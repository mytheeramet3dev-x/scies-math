# Time Series Analysis and Forecasting (`timeseries`)

The `timeseries` module provides parametric forecasting models (AR, MA, ARMA, ARIMA), exponential smoothing (Single, Holt, Holt-Winters with additive seasonality), rolling window statistics, and Dynamic Time Warping (DTW).

---

## 1. Autoregressive Integrated Moving Average (`ARIMA(p, d, q)`)

Models non-stationary time series $X_t$ by taking $d$-th differences $\nabla^d X_t = (1 - B)^d X_t$ to induce stationarity:

$$\left(1 - \sum_{i=1}^p \phi_i B^i\right) (1 - B)^d X_t = c + \left(1 + \sum_{j=1}^q \theta_j B^j\right) \varepsilon_t$$

where $B$ is the backshift lag operator $B^k X_t = X_{t-k}$, and $\varepsilon_t \sim \text{WN}(0, \sigma^2)$ is Gaussian white noise.

### Parameter Fitting & Forecasting
- `arima_fit(data, p, d, q)`: Differences series $d$ times, estimates autoregressive $\phi$ and moving average $\theta$ coefficients via Yule-Walker / conditional least squares.
- `arima_forecast(&model, steps)`: Generates multi-step forecasts with confidence intervals and inverse difference reconstruction.

---

## 2. Exponential Smoothing

### Holt-Winters Triple Exponential Smoothing (`holt_winters`)
Decomposes seasonal series with period $L$ into level $(\ell_t)$, trend $(b_t)$, and seasonal components $(s_t)$:

$$\ell_t = \alpha (y_t - s_{t-L}) + (1 - \alpha) (\ell_{t-1} + b_{t-1})$$

$$b_t = \beta (\ell_t - \ell_{t-1}) + (1 - \beta) b_{t-1}$$

$$s_t = \gamma (y_t - \ell_{t-1} - b_{t-1}) + (1 - \gamma) s_{t-L}$$

$$m\text{-step Forecast:} \quad \hat{y}_{t+m} = \ell_t + m b_t + s_{t - L + 1 + ((m-1) \bmod L)}$$

---

## 3. Dynamic Time Warping (`dtw_distance`)

Measures similarity between two temporal sequences $X = (x_1, \dots, x_N)$ and $Y = (y_1, \dots, y_M)$ that may vary in speed or duration:

$$D(i, j) = |x_i - y_j| + \min(D(i-1, j), D(i, j-1), D(i-1, j-1))$$

Computes the minimum cumulative cost alignment path in $O(N \cdot M)$ dynamic programming time.

---

## 4. Code Example

```rust
use scies_math_th::timeseries::{arima_fit, arima_forecast, dtw_distance, holt_winters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Fit ARIMA(1, 1, 1) model
    let series: Vec<f64> = (0..50).map(|t| (t as f64 * 0.5) + (t as f64 * 0.2).sin()).collect();
    let model = arima_fit(&series, 1, 1, 1)?;
    let forecast = arima_forecast(&model, 5)?;
    println!("Next 5 forecast steps: {:?}", forecast);

    // 2. Holt-Winters Seasonal Smoothing (Period = 4)
    let seasonal_data = vec![
        10.0, 20.0, 15.0, 5.0,
        12.0, 22.0, 17.0, 7.0,
        14.0, 24.0, 19.0, 9.0,
    ];
    let (fitted, future) = holt_winters(&seasonal_data, 4, 0.2, 0.1, 0.3, 4)?;
    println!("Seasonal 4-step forecast: {:?}", future);

    // 3. Dynamic Time Warping (DTW) distance
    let seq_a = vec![1.0, 2.0, 3.0, 4.0];
    let seq_b = vec![1.0, 1.5, 2.5, 3.5, 4.0];
    let d = dtw_distance(&seq_a, &seq_b);
    println!("DTW alignment distance: {:.4}", d);

    Ok(())
}
```
