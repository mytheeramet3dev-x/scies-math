use scies_math_th::units::{Dimension, Quantity};

#[test]
fn test_dimensional_physical_laws() {
    // Law of Universal Gravitation: F = G * m1 * m2 / r^2
    // Dimension of G: Force * Length^2 / Mass^2 = (M L T^-2) * L^2 / M^2 = M^-1 L^3 T^-2
    let dim_g = Dimension::FORCE * Dimension::LENGTH.powi(2) / Dimension::MASS.powi(2);

    let m1 = Quantity::kilograms(5.972e24); // Earth mass
    let m2 = Quantity::kilograms(7.348e22); // Moon mass
    let r = Quantity::meters(3.844e8); // Earth-Moon distance
    let g_const = Quantity::new(6.67430e-11, dim_g);

    let force = g_const * m1 * m2 / r.powi(2);
    let force_val = force.unwrap();

    assert_eq!(force_val.dimension, Dimension::FORCE);
    // Expected gravitational force ~ 1.98e20 N
    assert!((force_val.value - 1.982e20).abs() < 1e18);
}

#[test]
fn test_unit_conversions_and_errors() {
    let d1 = Quantity::kilometers(1.5);
    let d2 = Quantity::meters(500.0);

    let total = d1.add(&d2).unwrap();
    assert_eq!(total.value, 2000.0);
    assert_eq!(total.dimension, Dimension::LENGTH);
}
