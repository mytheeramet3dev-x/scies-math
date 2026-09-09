# Special Mathematical Functions (`special_functions`)

The `special_functions` module provides high-accuracy, zero-dependency approximations for transcendental and special functions commonly used in mathematical physics, statistical mechanics, and diffusion models.

---

## 1. Supported Function Families

### Error Function Family
- **Error Function**:
  $$\text{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt$$
- **Complementary Error Function**: $\text{erfc}(x) = 1 - \text{erf}(x)$ (evaluated with no catastrophic cancellation for large $x$).
- **Inverse Error Function**: $\text{erf}^{-1}(p)$ via rational Chebyshev approximations.

### Gamma and Beta Function Family
- **Gamma Function**:
  $$\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt, \quad \Gamma(n) = (n - 1)!$$
  Computed via the Lanczos approximation with 15-digit precision.
- **Log-Gamma**: $\ln \Gamma(x)$ for large arguments preventing IEEE 754 overflow.
- **Digamma Function**: $\psi(x) = \frac{d}{dx} \ln \Gamma(x) = \frac{\Gamma'(x)}{\Gamma(x)}$.
- **Beta Function**: $B(x, y) = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)} = \int_0^1 t^{x-1} (1-t)^{y-1} dt$.
- **Incomplete & Regularized Incomplete Beta**: $I_x(a, b) = \frac{B(x; a, b)}{B(a, b)}$ via continued fractions.
- **Lower Incomplete Gamma**: $\gamma(s, x) = \int_0^x t^{s-1} e^{-t} dt$ and $P(s, x) = \frac{\gamma(s, x)}{\Gamma(s)}$.

### Cylindrical Bessel Functions
- **Bessel $J_0(x)$ and $J_1(x)$**: Solutions of $x^2 y'' + x y' + (x^2 - \nu^2) y = 0$ of the first kind (order 0 and 1).
- **Bessel $Y_0(x)$ and $Y_1(x)$**: Solutions of the second kind (Weber functions).

---

## 2. Code Example

```rust
use scies_math_th::special_functions::{bessel_j0, erf, gamma, ln_gamma};

fn main() {
    // 1. Error function at x = 1.0 (Exact ~ 0.84270079)
    println!("erf(1.0) = {:.8}", erf(1.0));

    // 2. Factorial via Gamma: 5! = Gamma(6) = 120
    println!("Gamma(6.0) = {:.1}", gamma(6.0));

    // 3. Log-Gamma for large values (prevents overflow for 100!)
    println!("ln(Gamma(101)) = {:.4}", ln_gamma(101.0));

    // 4. Cylindrical Bessel function J_0(x) at first zero (x ≈ 2.4048)
    println!("J_0(2.404825) = {:.2e}", bessel_j0(2.4048255577));
}
```
