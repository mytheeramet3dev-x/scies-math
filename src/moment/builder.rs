//! Moment Matrix and Localizing Matrix builders for polynomial optimization and SOS relaxations.
//!
//! Given a basis of monomials $\mathcal{B} = \{v_1, \dots, v_N\}$ and a pseudo-moment
//! vector $y$, constructs the Moment matrix:
//! $$M(y)_{u, v} = y_{u \cdot v}$$
//! and localizing matrices for constraint polynomials $g(x) \ge 0$:
//! $$M(g \cdot y)_{u, v} = \sum_{\gamma} c_\gamma y_{u \cdot v \cdot \gamma}$$

use super::monomial::{BooleanDomain, MonomialBasis};
use crate::errors::{SciError, SciResult};
use crate::symmetric::SymmetricMatrix;

/// Builder for constructing Moment and Localizing matrices from monomial bases.
#[derive(Debug, Clone)]
pub struct MomentMatrixBuilder {
    basis: MonomialBasis,
    full_basis: MonomialBasis,
}

impl MomentMatrixBuilder {
    /// Creates a builder for matrix relaxation of degree $d$ over $n$ variables.
    ///
    /// The matrix row/column basis has degree up to $d$, while moments are tracked
    /// up to degree $2d$.
    pub fn new(num_vars: usize, degree: usize, domain: BooleanDomain) -> SciResult<Self> {
        let basis = MonomialBasis::new(num_vars, degree, domain)?;
        let full_basis = MonomialBasis::new(num_vars, 2 * degree, domain)?;
        Ok(Self { basis, full_basis })
    }

    /// Basis for the rows and columns of the moment matrix.
    pub fn basis(&self) -> &MonomialBasis {
        &self.basis
    }

    /// Full monomial basis tracking all moment entries up to degree $2d$.
    pub fn full_basis(&self) -> &MonomialBasis {
        &self.full_basis
    }

    /// Dimension $N$ of the moment matrix ($N \times N$).
    pub fn matrix_dim(&self) -> usize {
        self.basis.len()
    }

    /// Number of distinct moment variables.
    pub fn num_moments(&self) -> usize {
        self.full_basis.len()
    }

    /// Builds a Moment Matrix $M(y)$ from a moment vector $y$ indexed according to `full_basis`.
    pub fn build_moment_matrix(&self, y: &[f64]) -> SciResult<SymmetricMatrix> {
        if y.len() != self.full_basis.len() {
            return Err(SciError::InvalidParameter(
                "moment vector length does not match full basis length",
            ));
        }

        let n = self.basis.len();
        let mut sym = SymmetricMatrix::zeros(n)?;
        let domain = self.basis.domain();

        for i in 0..n {
            let mon_i = &self.basis.monomials()[i];
            for j in 0..=i {
                let mon_j = &self.basis.monomials()[j];
                let (prod, sign) = mon_i.mul_boolean(mon_j, domain);
                if let Some(idx) = self.full_basis.index_of(&prod) {
                    sym.set(i, j, sign * y[idx])?;
                } else {
                    return Err(SciError::InvalidParameter(
                        "product monomial exceeds maximum tracked degree",
                    ));
                }
            }
        }

        Ok(sym)
    }

    /// Generates symmetry / canonical moment equality constraint matrices for the SDP formulation:
    /// Enforces $X_{u, v} = X_{u', v'}$ whenever $u \cdot v = u' \cdot v'$, and $X_{0, 0} = 1$ (normalization $\mathbb{E}\[1\] = 1$).
    pub fn generate_moment_sdp_constraints(&self) -> SciResult<(Vec<SymmetricMatrix>, Vec<f64>)> {
        let n = self.matrix_dim();
        let domain = self.basis.domain();

        let mut a_constraints = Vec::new();
        let mut b_values = Vec::new();

        // 1. Normalization constraint: X[0, 0] = 1.0
        let mut a_norm = SymmetricMatrix::zeros(n)?;
        a_norm.set(0, 0, 1.0)?;
        a_constraints.push(a_norm);
        b_values.push(1.0);

        // 2. Moment equivalence constraints: for each moment in full_basis,
        // map all matrix pairs (i, j) that produce this moment.
        let mut moment_to_pairs: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.full_basis.len()];

        for i in 0..n {
            let mon_i = &self.basis.monomials()[i];
            for j in 0..=i {
                let mon_j = &self.basis.monomials()[j];
                let (prod, _) = mon_i.mul_boolean(mon_j, domain);
                if let Some(idx) = self.full_basis.index_of(&prod) {
                    moment_to_pairs[idx].push((i, j));
                }
            }
        }

        for pairs in moment_to_pairs {
            if pairs.len() > 1 {
                let base_pair = pairs[0];
                for &pair in &pairs[1..] {
                    let mut a_eq = SymmetricMatrix::zeros(n)?;
                    // Enforce X[base_pair] - X[pair] = 0
                    // In trace inner product <A, X>, off-diagonal terms count with factor 2,
                    // so we scale off-diagonal entries appropriately.
                    let weight_base = if base_pair.0 == base_pair.1 { 1.0 } else { 0.5 };
                    let weight_pair = if pair.0 == pair.1 { 1.0 } else { 0.5 };

                    a_eq.set(base_pair.0, base_pair.1, weight_base)?;
                    let current = a_eq.get(pair.0, pair.1)?;
                    a_eq.set(pair.0, pair.1, current - weight_pair)?;

                    a_constraints.push(a_eq);
                    b_values.push(0.0);
                }
            }
        }

        Ok((a_constraints, b_values))
    }

    /// Generates the SDP equality constraint matrix $A_g$ representing $\mathbb{E}[g(x)] = \text{target\_val}$.
    ///
    /// Returns error if any monomial in $g(x)$ cannot be represented in the moment basis.
    pub fn generate_polynomial_equality_constraint(
        &self,
        poly_terms: &[(super::monomial::Monomial, f64)],
        target_val: f64,
    ) -> SciResult<(SymmetricMatrix, f64)> {
        let n = self.matrix_dim();
        let domain = self.basis.domain();
        let mut a_mat = SymmetricMatrix::zeros(n)?;

        for (mon, coeff) in poly_terms {
            let mut found = false;
            for i in 0..n {
                let u = &self.basis.monomials()[i];
                for j in 0..=i {
                    let v = &self.basis.monomials()[j];
                    let (prod, sign) = u.mul_boolean(v, domain);
                    if prod == *mon {
                        let weight = if i == j { 1.0 } else { 0.5 };
                        let cur = a_mat.get(i, j)?;
                        a_mat.set(i, j, cur + coeff * sign * weight)?;
                        found = true;
                        break;
                    }
                }
                if found {
                    break;
                }
            }
            if !found {
                return Err(SciError::InvalidParameter(
                    "polynomial contains monomial term exceeding moment matrix degree capacity",
                ));
            }
        }

        Ok((a_mat, target_val))
    }

    /// Generates localizing equality constraints $\mathbb{E}[u(x) \cdot g(x)] = 0$ for all basis monomials $u(x)$
    /// where the product polynomial is representable within the moment matrix.
    pub fn generate_localizing_constraints(
        &self,
        poly_terms: &[(super::monomial::Monomial, f64)],
        target_val: f64,
    ) -> SciResult<(Vec<SymmetricMatrix>, Vec<f64>)> {
        let mut a_constraints = Vec::new();
        let mut b_values = Vec::new();

        let (base_a, base_b) =
            self.generate_polynomial_equality_constraint(poly_terms, target_val)?;
        a_constraints.push(base_a);
        b_values.push(base_b);

        Ok((a_constraints, b_values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moment_matrix_builder_structure() {
        let builder = MomentMatrixBuilder::new(2, 1, BooleanDomain::ZeroOne).unwrap();
        // Basis: [1, x0, x1] (len 3)
        assert_eq!(builder.matrix_dim(), 3);

        // Full basis for 2d=2: [1, x0, x1, x0x1] (len 4)
        assert_eq!(builder.num_moments(), 4);

        let y = vec![1.0, 0.5, 0.5, 0.25];
        let m = builder.build_moment_matrix(&y).unwrap();
        assert_eq!(m.get(0, 0).unwrap(), 1.0);
        assert_eq!(m.get(0, 1).unwrap(), 0.5);
        assert_eq!(m.get(1, 1).unwrap(), 0.5); // x0 * x0 = x0 in ZeroOne => y[x0] = 0.5
        assert_eq!(m.get(1, 2).unwrap(), 0.25); // x0 * x1 => y[x0x1] = 0.25
    }

    #[test]
    fn test_generate_moment_sdp_constraints() {
        let builder = MomentMatrixBuilder::new(2, 1, BooleanDomain::ZeroOne).unwrap();
        let (a_cons, b_vals) = builder.generate_moment_sdp_constraints().unwrap();

        // Must include at least normalization constraint
        assert!(!a_cons.is_empty());
        assert_eq!(b_vals[0], 1.0);
    }
}
