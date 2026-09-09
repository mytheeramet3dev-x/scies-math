//! Canonical expression simplification, constant folding, and identity rules.

use super::expr::Expr;
use crate::exact::Rational;

/// Recursively simplifies an expression using algebraic rewrite rules and constant folding.
pub fn simplify(expr: &Expr) -> Expr {
    match expr {
        Expr::Integer(_) | Expr::Rational(_) | Expr::Symbol(_) => expr.clone(),

        Expr::Neg(inner) => {
            let sim = simplify(inner);
            match sim {
                Expr::Integer(n) => Expr::Integer(-n),
                Expr::Rational(r) => Expr::Rational(-r),
                Expr::Neg(nested) => *nested,
                _ => Expr::Neg(Box::new(sim)),
            }
        }

        Expr::Add(terms) => {
            let mut flat_terms = Vec::new();
            for t in terms {
                let s = simplify(t);
                if let Expr::Add(nested) = s {
                    flat_terms.extend(nested);
                } else {
                    flat_terms.push(s);
                }
            }

            let mut const_rat = Rational::zero();
            let mut non_const_terms = Vec::new();

            for t in flat_terms {
                match t {
                    Expr::Integer(n) => const_rat = const_rat + Rational::from_integer(n),
                    Expr::Rational(r) => const_rat = const_rat + r,
                    _ => non_const_terms.push(t),
                }
            }

            // Group like terms (e.g. x + x -> 2*x)
            let mut grouped: Vec<(Expr, i128)> = Vec::new();
            for term in non_const_terms {
                if let Some(entry) = grouped.iter_mut().find(|(t, _)| *t == term) {
                    entry.1 += 1;
                } else {
                    grouped.push((term, 1));
                }
            }

            let mut final_terms = Vec::new();
            if const_rat != Rational::zero() || grouped.is_empty() {
                if const_rat.denom() == 1 {
                    final_terms.push(Expr::Integer(const_rat.numer()));
                } else {
                    final_terms.push(Expr::Rational(const_rat));
                }
            }

            for (term, count) in grouped {
                if count == 1 {
                    final_terms.push(term);
                } else {
                    final_terms.push(Expr::Mul(vec![Expr::Integer(count), term]));
                }
            }

            if final_terms.len() == 1 {
                final_terms.pop().unwrap()
            } else {
                Expr::Add(final_terms)
            }
        }

        Expr::Mul(factors) => {
            let mut flat_factors = Vec::new();
            for f in factors {
                let s = simplify(f);
                if let Expr::Mul(nested) = s {
                    flat_factors.extend(nested);
                } else {
                    flat_factors.push(s);
                }
            }

            let mut const_rat = Rational::one();
            let mut non_const_factors = Vec::new();

            for f in flat_factors {
                match f {
                    Expr::Integer(0) => return Expr::Integer(0),
                    Expr::Rational(r) if r == Rational::zero() => return Expr::Integer(0),
                    Expr::Integer(n) => const_rat = const_rat * Rational::from_integer(n),
                    Expr::Rational(r) => const_rat = const_rat * r,
                    _ => non_const_factors.push(f),
                }
            }

            let mut final_factors = Vec::new();
            if const_rat != Rational::one() || non_const_factors.is_empty() {
                if const_rat.denom() == 1 {
                    final_factors.push(Expr::Integer(const_rat.numer()));
                } else {
                    final_factors.push(Expr::Rational(const_rat));
                }
            }

            final_factors.extend(non_const_factors);

            if final_factors.is_empty() {
                Expr::Integer(1)
            } else if final_factors.len() == 1 {
                final_factors.pop().unwrap()
            } else {
                Expr::Mul(final_factors)
            }
        }

        Expr::Pow(base, exp) => {
            let b_sim = simplify(base);
            let e_sim = simplify(exp);

            match (&b_sim, &e_sim) {
                (_, Expr::Integer(0)) => Expr::Integer(1),
                (b, Expr::Integer(1)) => b.clone(),
                (Expr::Integer(0), _) => Expr::Integer(0),
                (Expr::Integer(1), _) => Expr::Integer(1),
                (Expr::Integer(b_val), Expr::Integer(e_val)) if *e_val >= 0 && *e_val <= 10 => {
                    Expr::Integer(b_val.pow(*e_val as u32))
                }
                _ => Expr::Pow(Box::new(b_sim), Box::new(e_sim)),
            }
        }

        Expr::Sin(inner) => Expr::Sin(Box::new(simplify(inner))),
        Expr::Cos(inner) => Expr::Cos(Box::new(simplify(inner))),
        Expr::Exp(inner) => {
            let s = simplify(inner);
            if let Expr::Integer(0) = s {
                Expr::Integer(1)
            } else {
                Expr::Exp(Box::new(s))
            }
        }
        Expr::Ln(inner) => {
            let s = simplify(inner);
            if let Expr::Integer(1) = s {
                Expr::Integer(0)
            } else {
                Expr::Ln(Box::new(s))
            }
        }
        Expr::Sqrt(inner) => Expr::Sqrt(Box::new(simplify(inner))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_basic_identities() {
        // x + 0 -> x
        let expr1 = Expr::Add(vec![Expr::sym("x"), Expr::int(0)]);
        assert_eq!(simplify(&expr1), Expr::sym("x"));

        // x * 1 -> x
        let expr2 = Expr::Mul(vec![Expr::sym("x"), Expr::int(1)]);
        assert_eq!(simplify(&expr2), Expr::sym("x"));

        // x * 0 -> 0
        let expr3 = Expr::Mul(vec![Expr::sym("x"), Expr::int(0)]);
        assert_eq!(simplify(&expr3), Expr::int(0));

        // x^0 -> 1
        let expr4 = Expr::Pow(Box::new(Expr::sym("x")), Box::new(Expr::int(0)));
        assert_eq!(simplify(&expr4), Expr::int(1));
    }

    #[test]
    fn test_simplify_constant_folding() {
        // 2 + 3 + x -> 5 + x
        let expr = Expr::Add(vec![Expr::int(2), Expr::int(3), Expr::sym("x")]);
        let sim = simplify(&expr);
        assert_eq!(sim, Expr::Add(vec![Expr::int(5), Expr::sym("x")]));
    }
}
