//! Symbolic expression tree (AST) representation and evaluation.

use crate::errors::{SciError, SciResult};
use crate::exact::Rational;
use core::fmt;
use std::collections::HashMap;

/// An abstract syntax tree (AST) node for mathematical expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    /// Exact integer literal.
    Integer(i128),
    /// Exact rational fraction literal $p/q$.
    Rational(Rational),
    /// Named variable or symbol (e.g. "x", "alpha").
    Symbol(String),
    /// Addition of multiple terms $e_1 + e_2 + \dots + e_k$.
    Add(Vec<Expr>),
    /// Multiplication of multiple factors $e_1 \cdot e_2 \cdots e_k$.
    Mul(Vec<Expr>),
    /// Exponentiation $base^{exp}$.
    Pow(Box<Expr>, Box<Expr>),
    /// Unary negation $-e$.
    Neg(Box<Expr>),
    /// Sine $\sin(e)$.
    Sin(Box<Expr>),
    /// Cosine $\cos(e)$.
    Cos(Box<Expr>),
    /// Exponential $\exp(e)$.
    Exp(Box<Expr>),
    /// Natural logarithm $\ln(e)$.
    Ln(Box<Expr>),
    /// Square root $\sqrt{e}$.
    Sqrt(Box<Expr>),
}

impl Expr {
    /// Creates an integer expression.
    pub const fn int(n: i128) -> Self {
        Self::Integer(n)
    }

    /// Creates a rational expression.
    pub const fn rational(r: Rational) -> Self {
        Self::Rational(r)
    }

    /// Creates a variable symbol expression.
    pub fn sym(name: impl Into<String>) -> Self {
        Self::Symbol(name.into())
    }

    /// Creates an addition expression $a + b$.
    #[allow(clippy::should_implement_trait)]
    pub fn add(a: Expr, b: Expr) -> Self {
        Self::Add(vec![a, b])
    }

    /// Creates a multiplication expression $a \cdot b$.
    #[allow(clippy::should_implement_trait)]
    pub fn mul(a: Expr, b: Expr) -> Self {
        Self::Mul(vec![a, b])
    }

    /// Creates an exponentiation expression $a^b$.
    pub fn pow(base: Expr, exp: Expr) -> Self {
        Self::Pow(Box::new(base), Box::new(exp))
    }

    /// Creates a negation $-a$.
    #[allow(clippy::should_implement_trait)]
    pub fn neg(a: Expr) -> Self {
        Self::Neg(Box::new(a))
    }

    /// Evaluates the expression numerically given a variable environment.
    pub fn eval(&self, vars: &HashMap<String, f64>) -> SciResult<f64> {
        match self {
            Self::Integer(n) => Ok(*n as f64),
            Self::Rational(r) => Ok(r.to_f64()),
            Self::Symbol(name) => vars.get(name).copied().ok_or_else(|| {
                SciError::SymbolicEvaluationError(format!("unbound symbol '{name}'"))
            }),
            Self::Add(terms) => {
                let mut sum = 0.0;
                for t in terms {
                    sum += t.eval(vars)?;
                }
                Ok(sum)
            }
            Self::Mul(factors) => {
                let mut prod = 1.0;
                for f in factors {
                    prod *= f.eval(vars)?;
                }
                Ok(prod)
            }
            Self::Pow(base, exp) => {
                let b = base.eval(vars)?;
                let e = exp.eval(vars)?;
                Ok(b.powf(e))
            }
            Self::Neg(inner) => Ok(-inner.eval(vars)?),
            Self::Sin(inner) => Ok(inner.eval(vars)?.sin()),
            Self::Cos(inner) => Ok(inner.eval(vars)?.cos()),
            Self::Exp(inner) => Ok(inner.eval(vars)?.exp()),
            Self::Ln(inner) => {
                let val = inner.eval(vars)?;
                if val <= 0.0 {
                    return Err(SciError::DomainError("ln of non-positive value"));
                }
                Ok(val.ln())
            }
            Self::Sqrt(inner) => {
                let val = inner.eval(vars)?;
                if val < 0.0 {
                    return Err(SciError::DomainError("sqrt of negative value"));
                }
                Ok(val.sqrt())
            }
        }
    }

    /// Substitutes all occurrences of variable `var_name` with `target`.
    pub fn substitute(&self, var_name: &str, target: &Expr) -> Expr {
        match self {
            Self::Symbol(s) if s == var_name => target.clone(),
            Self::Add(terms) => Self::Add(
                terms
                    .iter()
                    .map(|t| t.substitute(var_name, target))
                    .collect(),
            ),
            Self::Mul(factors) => Self::Mul(
                factors
                    .iter()
                    .map(|f| f.substitute(var_name, target))
                    .collect(),
            ),
            Self::Pow(b, e) => Self::Pow(
                Box::new(b.substitute(var_name, target)),
                Box::new(e.substitute(var_name, target)),
            ),
            Self::Neg(inner) => Self::Neg(Box::new(inner.substitute(var_name, target))),
            Self::Sin(inner) => Self::Sin(Box::new(inner.substitute(var_name, target))),
            Self::Cos(inner) => Self::Cos(Box::new(inner.substitute(var_name, target))),
            Self::Exp(inner) => Self::Exp(Box::new(inner.substitute(var_name, target))),
            Self::Ln(inner) => Self::Ln(Box::new(inner.substitute(var_name, target))),
            Self::Sqrt(inner) => Self::Sqrt(Box::new(inner.substitute(var_name, target))),
            _ => self.clone(),
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Integer(n) => write!(f, "{n}"),
            Self::Rational(r) => write!(f, "{r}"),
            Self::Symbol(s) => write!(f, "{s}"),
            Self::Add(terms) => {
                let formatted: Vec<String> = terms.iter().map(|t| format!("{t}")).collect();
                write!(f, "({})", formatted.join(" + "))
            }
            Self::Mul(factors) => {
                let formatted: Vec<String> = factors.iter().map(|t| format!("{t}")).collect();
                write!(f, "{}", formatted.join(" * "))
            }
            Self::Pow(b, e) => write!(f, "({b})^{e}"),
            Self::Neg(inner) => write!(f, "-({inner})"),
            Self::Sin(inner) => write!(f, "sin({inner})"),
            Self::Cos(inner) => write!(f, "cos({inner})"),
            Self::Exp(inner) => write!(f, "exp({inner})"),
            Self::Ln(inner) => write!(f, "ln({inner})"),
            Self::Sqrt(inner) => write!(f, "sqrt({inner})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_eval_and_substitute() {
        // f(x) = x^2 + 2x + 1
        let x = Expr::sym("x");
        let expr = Expr::Add(vec![
            Expr::Pow(Box::new(x.clone()), Box::new(Expr::int(2))),
            Expr::Mul(vec![Expr::int(2), x.clone()]),
            Expr::int(1),
        ]);

        let mut env = HashMap::new();
        env.insert("x".to_string(), 3.0);

        let val = expr.eval(&env).unwrap();
        // 3^2 + 2*3 + 1 = 16
        assert_eq!(val, 16.0);

        let sub = expr.substitute("x", &Expr::int(5));
        let val_sub = sub.eval(&HashMap::new()).unwrap();
        // 5^2 + 2*5 + 1 = 36
        assert_eq!(val_sub, 36.0);
    }
}
