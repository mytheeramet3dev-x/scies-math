# Uncertainty Propagation and Tolerance Policies (`uncertainty`, `policy`)

The `uncertainty` and `policy` modules implement first-order Gaussian error propagation (GUM standard) and centralized numerical tolerance policies.

---

## 1. Gaussian Error Propagation (`UncertainValue`)

Represents a measurement $x \pm u_x$ with standard uncertainty $u_x$:

- **Addition / Subtraction**: $u_{x \pm y} = \sqrt{u_x^2 + u_y^2}$
- **Multiplication**: $u_{x \cdot y} = |x \cdot y| \sqrt{\left(\frac{u_x}{x}\right)^2 + \left(\frac{u_y}{y}\right)^2}$
- **Division**: $u_{x / y} = \left|\frac{x}{y}\right| \sqrt{\left(\frac{u_x}{x}\right)^2 + \left(\frac{u_y}{y}\right)^2}$
- **Power $x^p$**: $u_{x^p} = |p \cdot x^{p-1}| u_x$

---

## 2. Tolerance Policies (`TolerancePolicy`)

Centralized precision configuration across numerical algorithms:
- `TolerancePolicy::Strict`: $10^{-10}$ tolerance (for high-assurance verified solvers).
- `TolerancePolicy::Standard`: $10^{-6}$ tolerance (general scientific default).
- `TolerancePolicy::Relaxed`: $10^{-3}$ tolerance (fast heuristics, real-time approximation).

---

## 3. Code Example

```rust
use scies_math_th::uncertainty::UncertainValue;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mass = UncertainValue::new(10.0, 0.1)?; // 10.0 +/- 0.1 kg
    let volume = UncertainValue::new(2.0, 0.05)?; // 2.0 +/- 0.05 m^3

    // Density: rho = m / V
    let density = (mass / volume)?;
    println!("Density: {:.2} +/- {:.2}", density.value(), density.uncertainty()); // 5.00 +/- 0.14

    Ok(())
}
```
