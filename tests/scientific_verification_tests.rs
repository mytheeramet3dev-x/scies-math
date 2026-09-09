//! Cross-verification tests for optimization, autodiff, ODEs, and statistical distributions.

use scies_math_th::autodiff::{Dual, jacobian};
use scies_math_th::ode::rk4_system;
use scies_math_th::opt_multivar::bfgs;
use scies_math_th::probability::normal_pdf;

#[test]
fn test_bfgs_rosenbrock_convergence() {
    // 2D Rosenbrock function: f(x, y) = (1 - x)^2 + 100(y - x^2)^2
    // Global minimum at (1, 1) with f(1, 1) = 0
    let rosenbrock = |v: &[Dual]| -> Dual {
        let x = v[0];
        let y = v[1];
        let one = Dual::from(1.0);
        let hundred = Dual::from(100.0);
        (one - x) * (one - x) + hundred * (y - x * x) * (y - x * x)
    };

    let x0 = [-1.2, 1.0];
    let (x_min, f_min) = bfgs(rosenbrock, &x0, 1e-6, 1000).unwrap();

    assert!((x_min[0] - 1.0).abs() < 1e-4, "x_min[0] = {}", x_min[0]);
    assert!((x_min[1] - 1.0).abs() < 1e-4, "x_min[1] = {}", x_min[1]);
    assert!(f_min < 1e-6, "f_min = {f_min}");
}

#[test]
fn test_autodiff_vs_analytic_gradient() {
    // f(x, y) = x^2 * y + sin(x)
    // df/dx = 2*x*y + cos(x)
    // df/dy = x^2
    let f = |v: &[Dual]| -> Dual {
        let x = v[0];
        let y = v[1];
        x * x * y + x.sin()
    };

    let pt = [1.5_f64, 2.0_f64];
    let ad_grad = jacobian(f, &pt);

    let analytic_df_dx = 2.0 * pt[0] * pt[1] + pt[0].cos();
    let analytic_df_dy = pt[0] * pt[0];

    assert!((ad_grad[0] - analytic_df_dx).abs() < 1e-12);
    assert!((ad_grad[1] - analytic_df_dy).abs() < 1e-12);
}

#[test]
fn test_rk4_harmonic_oscillator() {
    // System: y'' + y = 0 => y1' = y2, y2' = -y1
    // Initial condition: y1(0) = 1, y2(0) = 0
    // Analytic solution: y1(t) = cos(t)
    let f = |_t: f64, y: &[f64]| -> Vec<f64> { vec![y[1], -y[0]] };

    let y0 = vec![1.0, 0.0];
    let t0 = 0.0;
    let t_end = std::f64::consts::PI;
    let dt = 0.01;

    let sol = rk4_system(f, &y0, t0, t_end, dt).unwrap();
    let y_final = sol.final_state().unwrap();

    // At t = pi, cos(pi) = -1.0
    assert!(
        (y_final[0] - (-1.0)).abs() < 1e-4,
        "y_final[0] = {}",
        y_final[0]
    );
}

#[test]
fn test_normal_distribution_invariants() {
    // PDF symmetry around mean
    let mu = 2.5;
    let sigma = 1.2;

    let pdf_left = normal_pdf(mu - 1.0, mu, sigma).unwrap();
    let pdf_right = normal_pdf(mu + 1.0, mu, sigma).unwrap();
    assert!((pdf_left - pdf_right).abs() < 1e-14);

    // Invalid sigma rejection
    assert!(normal_pdf(0.0, 0.0, -1.0).is_err());
}
