use scies_math_th::exact::{Interval, Rational};

#[test]
fn test_exact_rational_algebraic_identities() {
    let a = Rational::new(3, 7).unwrap();
    let b = Rational::new(5, 11).unwrap();
    let c = Rational::new(2, 13).unwrap();

    // Associativity of addition: (a + b) + c == a + (b + c)
    assert_eq!((a + b) + c, a + (b + c));

    // Distributivity: a * (b + c) == (a * b) + (a * c)
    assert_eq!(a * (b + c), (a * b) + (a * c));

    // Inverse: a * a^{-1} == 1
    assert_eq!(a * a.recip().unwrap(), Rational::one());
}

#[test]
fn test_interval_kahan_bounds() {
    let x = Interval::new(2.0, 3.0).unwrap();
    let y = Interval::new(4.0, 5.0).unwrap();

    let prod = x * y;
    assert_eq!(prod.lower(), 8.0);
    assert_eq!(prod.upper(), 15.0);

    let div = (y / x).unwrap();
    assert_eq!(div.lower(), 4.0 / 3.0);
    assert_eq!(div.upper(), 5.0 / 2.0);
}
