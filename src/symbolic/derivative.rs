//! Exact symbolic differentiation and calculus operations.

use super::expr::Expr;
use super::simplify::simplify;

/// Computes the exact symbolic derivative $\frac{d}{d(\text{var})}[\text{expr}]$ and simplifies the result.
pub fn diff(expr: &Expr, var: &str) -> Expr {
    let raw = diff_raw(expr, var);
    simplify(&raw)
}

fn diff_raw(expr: &Expr, var: &str) -> Expr {
    match expr {
        Expr::Integer(_) | Expr::Rational(_) => Expr::Integer(0),

        Expr::Symbol(s) => {
            if s == var {
                Expr::Integer(1)
            } else {
                Expr::Integer(0)
            }
        }

        Expr::Neg(inner) => Expr::Neg(Box::new(diff_raw(inner, var))),

        Expr::Add(terms) => {
            let diff_terms: Vec<Expr> = terms.iter().map(|t| diff_raw(t, var)).collect();
            Expr::Add(diff_terms)
        }

        Expr::Mul(factors) => {
            // General Product Rule: d/dx [f1 * ... * fn] = sum_i (f_i' * prod_{j != i} f_j)
            let mut sum_terms = Vec::new();
            for i in 0..factors.len() {
                let mut term_factors = Vec::new();
                for (j, f) in factors.iter().enumerate() {
                    if i == j {
                        term_factors.push(diff_raw(f, var));
                    } else {
                        term_factors.push(f.clone());
                    }
                }
                sum_terms.push(Expr::Mul(term_factors));
            }
            Expr::Add(sum_terms)
        }

        Expr::Pow(base, exp) => {
            // Check if exp is constant integer n
            if let Expr::Integer(n) = **exp {
                if n == 0 {
                    Expr::Integer(0)
                } else if n == 1 {
                    diff_raw(base, var)
                } else {
                    // d/dx [u^n] = n * u^(n-1) * u'
                    let du = diff_raw(base, var);
                    Expr::Mul(vec![
                        Expr::Integer(n),
                        Expr::Pow(base.clone(), Box::new(Expr::Integer(n - 1))),
                        du,
                    ])
                }
            } else {
                // General d/dx [u^v] = u^v * (v' * ln(u) + v * u' / u)
                let du = diff_raw(base, var);
                let dv = diff_raw(exp, var);
                Expr::Mul(vec![
                    Expr::Pow(base.clone(), exp.clone()),
                    Expr::Add(vec![
                        Expr::Mul(vec![dv, Expr::Ln(base.clone())]),
                        Expr::Mul(vec![
                            *exp.clone(),
                            du,
                            Expr::Pow(base.clone(), Box::new(Expr::Integer(-1))),
                        ]),
                    ]),
                ])
            }
        }

        Expr::Sin(inner) => {
            // d/dx [sin(u)] = cos(u) * u'
            let du = diff_raw(inner, var);
            Expr::Mul(vec![Expr::Cos(inner.clone()), du])
        }

        Expr::Cos(inner) => {
            // d/dx [cos(u)] = -sin(u) * u'
            let du = diff_raw(inner, var);
            Expr::Mul(vec![Expr::Neg(Box::new(Expr::Sin(inner.clone()))), du])
        }

        Expr::Exp(inner) => {
            // d/dx [exp(u)] = exp(u) * u'
            let du = diff_raw(inner, var);
            Expr::Mul(vec![Expr::Exp(inner.clone()), du])
        }

        Expr::Ln(inner) => {
            // d/dx [ln(u)] = u' / u = u' * u^(-1)
            let du = diff_raw(inner, var);
            Expr::Mul(vec![
                du,
                Expr::Pow(inner.clone(), Box::new(Expr::Integer(-1))),
            ])
        }

        Expr::Sqrt(inner) => {
            // d/dx [sqrt(u)] = u' / (2 * sqrt(u))
            let du = diff_raw(inner, var);
            Expr::Mul(vec![
                Expr::Rational(crate::exact::Rational::new(1, 2).unwrap()),
                du,
                Expr::Pow(
                    Box::new(Expr::Sqrt(inner.clone())),
                    Box::new(Expr::Integer(-1)),
                ),
            ])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_symbolic_derivatives() {
        // f(x) = x^3 + 2x
        let x = Expr::sym("x");
        let f = Expr::Add(vec![
            Expr::Pow(Box::new(x.clone()), Box::new(Expr::int(3))),
            Expr::Mul(vec![Expr::int(2), x.clone()]),
        ]);

        // f'(x) = 3x^2 + 2
        let df = diff(&f, "x");

        let mut env = HashMap::new();
        env.insert("x".to_string(), 2.0);

        // f'(2) = 3*(4) + 2 = 14
        let val = df.eval(&env).unwrap();
        assert_eq!(val, 14.0);
    }

    #[test]
    fn test_trig_derivative() {
        // f(x) = sin(x) -> f'(x) = cos(x)
        let x = Expr::sym("x");
        let f = Expr::Sin(Box::new(x));
        let df = diff(&f, "x");

        let mut env = HashMap::new();
        env.insert("x".to_string(), 0.0);
        // cos(0) = 1
        assert_eq!(df.eval(&env).unwrap(), 1.0);
    }
}
