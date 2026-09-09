# Ordinary Differential Equation Solvers (`ode`)

The `ode` module provides foundational fixed-step and adaptive numerical integration schemes for initial value problems (IVPs):

$$\frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y}), \quad \mathbf{y}(t_0) = \mathbf{y}_0$$

---

## 1. Classical Solvers

| Method | Type | Order | Error per Step | Best For |
| :--- | :--- | :--- | :--- | :--- |
| `euler` / `euler_system` | Explicit Forward Euler | 1 | $O(\Delta t^2)$ | Quick qualitative checks, pedagogical demos |
| `rk4` / `rk4_system` | Classical 4th-order Runge-Kutta | 4 | $O(\Delta t^5)$ | Fixed step-size scientific integration |
| `rk45` | Adaptive Dormand-Prince | 4(5) | Adaptive | General scientific computing with automatic error control |

---

## 2. Adaptive Step Control: Dormand-Prince 4(5) (`rk45`)

Computes both 4th-order and 5th-order solutions simultaneously using 6 function evaluations:

$$y_{n+1} = y_n + \sum_{i=1}^6 b_i k_i \quad (\text{Order 5})$$

$$\hat{y}_{n+1} = y_n + \sum_{i=1}^6 \hat{b}_i k_i \quad (\text{Order 4})$$

Local truncation error estimate:

$$e_{n+1} = \|y_{n+1} - \hat{y}_{n+1}\|_2$$

Optimal next step size:

$$\Delta t_{\text{next}} = \Delta t \cdot \min\left(2.0, \max\left(0.2, 0.9 \left(\frac{\text{tol}}{e_{n+1}}\right)^{1/5}\right)\right)$$

where $\text{tol} = \text{atol} + \text{rtol} \cdot \max(\|y_n\|, \|y_{n+1}\|)$.

```rust
use scies_math_th::ode::rk45;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 2D Harmonic Oscillator: y0' = y1, y1' = -y0
    let harmonic = |_t: f64, y: &[f64]| vec![y[1], -y[0]];

    let sol = rk45(
        harmonic,
        0.0, std::f64::consts::PI, // t in [0, pi]
        vec![1.0, 0.0],            // y(0) = 1.0, y'(0) = 0.0
        1e-7, 1e-10,               // rtol, atol
        0.1,                       // dt_init
    )?;

    let final_y = sol.y.last().unwrap();
    println!("y(pi) = {:.6} (Exact: -1.0)", final_y[0]);
    println!("y'(pi) = {:.6} (Exact: 0.0)", final_y[1]);

    Ok(())
}
```
