# `timeseries` Module Documentation

Time series analysis — ARIMA, exponential smoothing, DTW, rolling statistics.

# Models

| Model | Function | Description |
|---|---|---|
| AR(p) | `ar_fit` / `ar_forecast` | Autoregressive |
| MA(q) | `ma_fit` / `ma_forecast` | Moving-average |
| ARMA(p,q) | `arma_fit` / `arma_forecast` | Combined AR + MA |
| ARIMA(p,d,q) | `arima_fit` / `arima_forecast` | With differencing |
| Simple ES | `simple_exp_smoothing` | Single exponential smoothing (SES) |
| Holt | `holt_linear` | Double ES with trend |
| Holt-Winters | `holt_winters` | Triple ES with trend + seasonality |
| DTW | `dtw_distance` | Dynamic Time Warping distance |

# Rolling statistics

| Function | Description |
|---|---|
| `rolling_mean(data, w)` | Window average |
| `rolling_std(data, w)` | Window standard deviation |
| `rolling_min` / `rolling_max` | Window extremes |
| `ewm(data, alpha)` | Exponentially-weighted mean |

# Usage

## ARIMA

```rust
use scies_math::timeseries::{arima_fit, arima_forecast};

let data: Vec<f64> = (0..100).map(|i| i as f64 + (i as f64 * 0.1).sin()).collect();
let model = arima_fit(&data, 1, 1, 1).unwrap(); // p=1, d=1, q=1
let future = arima_forecast(&model, 10);        // 10 steps ahead
```

## Holt-Winters (additive seasonality)

```rust
use scies_math::timeseries::holt_winters;

let seasonal_data: Vec<f64> = (0..48).map(|i| {
    10.0 + (i as f64 * 0.1) + 5.0 * (i as f64 * std::f64::consts::PI / 6.0).sin()
}).collect();
let (fitted, forecast) = holt_winters(&seasonal_data, 12, 0.2, 0.1, 0.3, 12).unwrap();
```

## Dynamic Time Warping

DTW finds the optimal alignment between two time series of potentially
different lengths by allowing time-axis warping.

```rust
use scies_math::timeseries::dtw_distance;

let a = vec![1.0, 2.0, 3.0, 4.0];
let b = vec![1.0, 2.0, 2.5, 3.5, 4.0]; // different length
let d = dtw_distance(&a, &b);
```