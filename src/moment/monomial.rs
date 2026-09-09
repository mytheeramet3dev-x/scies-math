//! Canonical monomial indexing and Boolean quotient reductions.
//!
//! Provides graded reverse lexicographic ordering for multi-indices and
//! algebraic quotient operations for Boolean variables ($x_i^2 = x_i$ or $x_i^2 = 1$).

use crate::errors::SciResult;
use std::cmp::Ordering;

/// Boolean domain representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanDomain {
    /// Zero-one variables $x_i \in \{0, 1\}$ satisfying idempotence $x_i^2 = x_i$.
    ZeroOne,
    /// Plus-minus-one spin variables $s_i \in \{-1, +1\}$ satisfying reflection $s_i^2 = 1$.
    PlusMinusOne,
}

/// A monomial represented by the exponents of variables $x_0, x_1, \dots, x_{n-1}$.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Monomial {
    /// Non-zero variable indices and their positive exponents: `(var_index, exponent)`.
    /// Maintained in strictly ascending order of `var_index`.
    powers: Vec<(usize, usize)>,
}

impl Monomial {
    /// Creates the constant monomial $1$ (degree 0).
    pub fn constant() -> Self {
        Self { powers: vec![] }
    }

    /// Creates a single variable monomial $x_i^1$.
    pub fn single_var(var: usize) -> Self {
        Self {
            powers: vec![(var, 1)],
        }
    }

    /// Creates a monomial from a list of `(var_index, exponent)` pairs.
    pub fn from_powers(mut powers: Vec<(usize, usize)>) -> Self {
        powers.retain(|&(_, exp)| exp > 0);
        powers.sort_unstable_by_key(|&(var, _)| var);

        let mut consolidated: Vec<(usize, usize)> = Vec::with_capacity(powers.len());
        for (var, exp) in powers {
            if let Some(last) = consolidated.last_mut().filter(|last| last.0 == var) {
                last.1 += exp;
            } else {
                consolidated.push((var, exp));
            }
        }
        consolidated.retain(|&(_, exp)| exp > 0);
        Self {
            powers: consolidated,
        }
    }

    /// Total degree $\sum_i a_i$ of the monomial.
    pub fn degree(&self) -> usize {
        self.powers.iter().map(|&(_, exp)| exp).sum()
    }

    /// Number of distinct variables appearing in the monomial.
    pub fn num_vars(&self) -> usize {
        self.powers.len()
    }

    /// List of `(var_index, exponent)` pairs.
    pub fn powers(&self) -> &[(usize, usize)] {
        &self.powers
    }

    /// Multiplies two monomials under polynomial ring $\mathbb{R}[x_1, \dots, x_n]$.
    pub fn mul(&self, other: &Self) -> Self {
        let mut combined = self.powers.clone();
        combined.extend_from_slice(&other.powers);
        Self::from_powers(combined)
    }

    /// Multiplies two monomials modulo the Boolean quotient ideal.
    pub fn mul_boolean(&self, other: &Self, domain: BooleanDomain) -> (Self, f64) {
        let raw = self.mul(other);
        match domain {
            BooleanDomain::ZeroOne => {
                // In {0, 1}, x_i^k = x_i for any k >= 1
                let reduced_powers: Vec<(usize, usize)> = raw
                    .powers
                    .into_iter()
                    .map(|(var, exp)| (var, if exp > 0 { 1 } else { 0 }))
                    .collect();
                (Self::from_powers(reduced_powers), 1.0)
            }
            BooleanDomain::PlusMinusOne => {
                // In {-1, +1}, x_i^2 = 1, so x_i^k = x_i if k is odd, and 1 if k is even.
                let reduced_powers: Vec<(usize, usize)> = raw
                    .powers
                    .into_iter()
                    .map(|(var, exp)| (var, exp % 2))
                    .filter(|&(_, exp)| exp > 0)
                    .collect();
                (Self::from_powers(reduced_powers), 1.0)
            }
        }
    }
}

impl PartialOrd for Monomial {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Monomial {
    /// Graded reverse lexicographic order (DegRevLex).
    fn cmp(&self, other: &Self) -> Ordering {
        let deg_cmp = self.degree().cmp(&other.degree());
        if deg_cmp != Ordering::Equal {
            return deg_cmp;
        }

        // For equal degree, compare reverse lexicographically
        let max_var = self
            .powers
            .iter()
            .chain(other.powers.iter())
            .map(|&(v, _)| v)
            .max()
            .unwrap_or(0);

        for v in (0..=max_var).rev() {
            let exp_self = self
                .powers
                .iter()
                .find(|&&(var, _)| var == v)
                .map(|&(_, exp)| exp)
                .unwrap_or(0);
            let exp_other = other
                .powers
                .iter()
                .find(|&&(var, _)| var == v)
                .map(|&(_, exp)| exp)
                .unwrap_or(0);

            if exp_self != exp_other {
                // In RevLex, smaller exponent in higher variable comes first (greater)
                return exp_other.cmp(&exp_self);
            }
        }

        Ordering::Equal
    }
}

/// Monomial basis up to degree $d$ over $n$ variables.
#[derive(Debug, Clone)]
pub struct MonomialBasis {
    num_vars: usize,
    max_degree: usize,
    domain: BooleanDomain,
    monomials: Vec<Monomial>,
}

impl MonomialBasis {
    /// Generates the canonical monomial basis for $n$ variables up to degree $d$.
    pub fn new(num_vars: usize, max_degree: usize, domain: BooleanDomain) -> SciResult<Self> {
        if max_degree == 0 {
            return Ok(Self {
                num_vars,
                max_degree,
                domain,
                monomials: vec![Monomial::constant()],
            });
        }

        let mut monomials = Vec::new();
        // Generate all subsets of variables of size <= max_degree (for multilinear / Boolean basis)
        generate_monomials_recursive(num_vars, max_degree, 0, &mut vec![], &mut monomials);

        // Sort canonically
        monomials.sort();

        Ok(Self {
            num_vars,
            max_degree,
            domain,
            monomials,
        })
    }

    /// Number of monomials in the basis.
    pub fn len(&self) -> usize {
        self.monomials.len()
    }

    /// Whether the basis is empty.
    pub fn is_empty(&self) -> bool {
        self.monomials.is_empty()
    }

    /// Slice of all monomials in canonical order.
    pub fn monomials(&self) -> &[Monomial] {
        &self.monomials
    }

    /// Finds index of a monomial in the basis.
    pub fn index_of(&self, mon: &Monomial) -> Option<usize> {
        self.monomials.binary_search(mon).ok()
    }

    /// Boolean domain of the basis.
    pub fn domain(&self) -> BooleanDomain {
        self.domain
    }

    /// Number of variables in the basis.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Maximum degree of monomials in the basis.
    pub fn max_degree(&self) -> usize {
        self.max_degree
    }
}

fn generate_monomials_recursive(
    num_vars: usize,
    max_degree: usize,
    start_var: usize,
    current: &mut Vec<usize>,
    out: &mut Vec<Monomial>,
) {
    // Convert current to Monomial
    let powers: Vec<(usize, usize)> = current.iter().map(|&v| (v, 1)).collect();
    out.push(Monomial::from_powers(powers));

    if current.len() >= max_degree {
        return;
    }

    for v in start_var..num_vars {
        current.push(v);
        generate_monomials_recursive(num_vars, max_degree, v + 1, current, out);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monomial_multiplication_zero_one() {
        let x0 = Monomial::single_var(0);
        let x1 = Monomial::single_var(1);

        // x0 * x0 = x0 in ZeroOne domain
        let (prod1, coeff1) = x0.mul_boolean(&x0, BooleanDomain::ZeroOne);
        assert_eq!(prod1, x0);
        assert_eq!(coeff1, 1.0);

        // x0 * x1 = x0 x1
        let (prod2, _) = x0.mul_boolean(&x1, BooleanDomain::ZeroOne);
        assert_eq!(prod2.degree(), 2);
    }

    #[test]
    fn test_monomial_basis_generation() {
        // n = 3 variables {0, 1, 2}, max_degree = 2
        // Monomials: 1 (deg 0), x0, x1, x2 (deg 1), x0x1, x0x2, x1x2 (deg 2) => total 1 + 3 + 3 = 7
        let basis = MonomialBasis::new(3, 2, BooleanDomain::ZeroOne).unwrap();
        assert_eq!(basis.len(), 7);
        assert_eq!(basis.monomials()[0], Monomial::constant());
    }
}
