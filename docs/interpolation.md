# Interpolation and Approximation (`interpolation`)

The `interpolation` module provides 1D and 2D function approximation routines, smooth piecewise polynomial splines, and Radial Basis Function (RBF) interpolation for scattered multidimensional datasets.

---

## 1. Interpolation Method Selection

| Method | Continuity | Oscillations / Runge's Phenomenon | Complexity | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- |
| `linear_interp` | $C^0$ | None | $O(\log N)$ search | Dense lookup tables, fast piecewise evaluation |
| `lagrange_interp`| $C^\infty$ | Severe for $N > 10$ | $O(N^2)$ | Analytical low-degree polynomial approximations |
| `neville_interp` | $C^\infty$ | Moderate | $O(N^2)$ recursive | Stable evaluation of polynomial interpolants |
| `cubic_spline` | $C^2$ | Minimal | $O(N)$ tridiagonal solve | Natural smooth interpolation, computer graphics |
| `akima_spline` | $C^1$ | **Zero overshoot** | $O(N)$ local slopes | Data with sudden step discontinuities or outliers |
| `bilinear_interp`| $C^0$ | None | $O(1)$ per cell | 2D rectangular grid interpolation (images, heatmaps) |
| `rbf_interp` | $C^\infty$ / $C^2$ | Controlled via kernel shape | $O(N^3)$ dense solve | High-dimensional scattered / unstructured data |

---

## 2. Cubic Spline vs. Akima Spline

### Natural Cubic Spline (`cubic_spline`)
Constructs piecewise cubic polynomials $S_i(x) = a_i + b_i(x - x_i) + c_i(x - x_i)^2 + d_i(x - x_i)^3$ on each interval $[x_i, x_{i+1}]$ ensuring:
1. $S_i(x_i) = y_i, \quad S_i(x_{i+1}) = y_{i+1}$
2. $S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})$ ($C^1$ continuity)
3. $S''_i(x_{i+1}) = S''_{i+1}(x_{i+1})$ ($C^2$ continuity)
4. Natural boundary conditions: $S''_0(x_0) = S''_{n-1}(x_n) = 0$.

Solves the resulting symmetric tridiagonal system in $O(N)$ via the Thomas algorithm.

### Akima Spline (`akima_spline`)
Estimates slopes $t_i$ using local 5-point weighted differences:

$$t_i = \frac{|m_{i+1} - m_i| m_{i-1} + |m_{i-1} - m_{i-2}| m_i}{|m_{i+1} - m_i| + |m_{i-1} - m_{i-2}|}$$

where $m_i = \frac{y_{i+1} - y_i}{x_{i+1} - x_i}$. This locally adapts to steep gradients and completely eliminates the unphysical "wiggles" seen in natural splines near step functions.

---

## 3. Radial Basis Functions (`rbf_interp`)

Interpolates scattered points $\{\mathbf{x}_i, y_i\}_{i=1}^N$ via linear combinations of radially symmetric kernel functions $\phi(r)$:

$$f(\mathbf{x}) = \sum_{j=1}^N w_j \phi(\|\mathbf{x} - \mathbf{x}_j\|)$$

Supported kernels:
- **Gaussian**: $\phi(r) = \exp(-(\epsilon r)^2)$
- **Multiquadric**: $\phi(r) = \sqrt{1 + (\epsilon r)^2}$
- **Inverse Multiquadric**: $\phi(r) = \frac{1}{\sqrt{1 + (\epsilon r)^2}}$
- **Thin Plate Spline**: $\phi(r) = r^2 \ln(r)$

---

## 4. Code Example

```rust
use scies_math_th::interpolation::{akima_spline, cubic_spline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let ys = vec![0.0, 0.0, 10.0, 10.0, 0.0]; // Step-like impulse data

    // Natural Cubic Spline (smoothest, but exhibits overshoot)
    let c_spline = cubic_spline(&xs, &ys)?;
    println!("Cubic spline at x=1.5: {:.3}", c_spline.eval(1.5)?);

    // Akima Spline (resists oscillation around the step at x=2)
    let a_spline = akima_spline(&xs, &ys)?;
    println!("Akima spline at x=1.5: {:.3}", a_spline.eval(1.5)?);

    Ok(())
}
```
