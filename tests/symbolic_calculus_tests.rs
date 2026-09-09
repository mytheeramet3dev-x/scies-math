use scies_math_th::symbolic::{Expr, Polynomial, diff, simplify};
use std::collections::HashMap;

#[test]
fn test_symbolic_polynomial_derivative_and_chain_rule() {
    // f(x) = exp(x^2)
    let x = Expr::sym("x");
    let f = Expr::Exp(Box::new(Expr::Pow(
        Box::new(x.clone()),
        Box::new(Expr::int(2)),
    )));

    // f'(x) = 2x * exp(x^2)
    let df = diff(&f, "x");

    let mut env = HashMap::new();
    env.insert("x".to_string(), 1.0);

    // f'(1) = 2(1) * exp(1) = 2e approx 5.43656
    let val = df.eval(&env).unwrap();
    assert!((val - 2.0 * std::f64::consts::E).abs() < 1e-6);
}

#[test]
fn test_polynomial_algebra_roundtrip() {
    // P(x) = 3x^2 + 2x + 1
    let p = Polynomial::new(vec![1.0, 2.0, 3.0]);
    let expr = p.to_expr("x");
    let simplified = simplify(&expr);

    let mut env = HashMap::new();
    env.insert("x".to_string(), 2.0);

    // P(2) = 3(4) + 2(2) + 1 = 17
    assert_eq!(p.eval(2.0), 17.0);
    assert_eq!(simplified.eval(&env).unwrap(), 17.0);
}
