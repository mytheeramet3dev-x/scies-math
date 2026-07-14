# `signal` Module Documentation

Signal processing — FFT, convolution, and digital filters.

# Functions

| Function | Description |
|---|---|
| `fft(signal)` | Radix-2 Cooley-Tukey FFT (length must be power of 2) |
| `ifft(spectrum)` | Inverse FFT |
| `fft_freq(n, dt)` | Frequency bins for FFT output |
| `power_spectrum(signal)` | |X(f)|² |
| `convolve(a, b)` | Linear convolution via direct sum |
| `correlate(a, b)` | Cross-correlation |
| `fir_filter(signal, coeffs)` | FIR filter (direct form) |
| `iir_filter(signal, b, a)` | IIR filter (direct form II) |

# Usage

```rust
use scies_math_th::signal::{fft, fft_freq, power_spectrum};

// 440 Hz sine wave, sample rate 44100 Hz
let n = 4096;
let dt = 1.0 / 44100.0;
let signal: Vec<f64> = (0..n)
    .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 * dt).sin())
    .collect();

let spectrum = fft(&signal);
let freqs    = fft_freq(n, dt);
let power    = power_spectrum(&signal);
// Peak in power at index ≈ 440 * n * dt ≈ 43
```

For STFT, wavelets, Hilbert transform, and adaptive filters see [`crate::signal_ext`].
