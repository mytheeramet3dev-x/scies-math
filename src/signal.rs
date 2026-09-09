//! Signal processing — FFT, convolution, and digital filters.
//!
//! # Functions
//!
//! | Function | Description |
//! |---|---|
//! | `fft(signal)` | Radix-2 Cooley-Tukey FFT (length must be power of 2) |
//! | `ifft(spectrum)` | Inverse FFT |
//! | `fft_freq(n, dt)` | Frequency bins for FFT output |
//! | `power_spectrum(signal)` | $\|X(f)\|^2$ |
//! | `convolve(a, b)` | Linear convolution via direct sum |
//! | `correlate(a, b)` | Cross-correlation |
//! | `fir_filter(signal, coeffs)` | FIR filter (direct form) |
//! | `iir_filter(signal, b, a)` | IIR filter (direct form II) |
//!
//! # Usage
//!
//! ```ignore
//! use scies_math_th::signal::{fft, fft_freq, power_spectrum};
//!
//! // 440 Hz sine wave, sample rate 44100 Hz
//! let n = 4096;
//! let dt = 1.0 / 44100.0;
//! let signal: Vec<f64> = (0..n)
//!     .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 * dt).sin())
//!     .collect();
//!
//! let spectrum = fft(&signal);
//! let freqs    = fft_freq(n, dt);
//! let power    = power_spectrum(&signal);
//! // Peak in power at index ≈ 440 * n * dt ≈ 43
//! ```
//!
//! For STFT, wavelets, Hilbert transform, and adaptive filters see [`crate::signal_ext`].
use crate::errors::{SciError, SciResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn magnitude(self) -> f64 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    pub fn conjugate(self) -> Self {
        Self::new(self.re, -self.im)
    }

    fn exp_i(theta: f64) -> Self {
        Self::new(theta.cos(), theta.sin())
    }
}

impl core::ops::Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl core::ops::Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl core::ops::Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re * rhs.re - self.im * rhs.im,
            self.re * rhs.im + self.im * rhs.re,
        )
    }
}

impl core::ops::Mul<f64> for Complex {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl core::ops::Div<f64> for Complex {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

pub fn fft(input: &[Complex]) -> SciResult<Vec<Complex>> {
    if input.is_empty() {
        return Err(SciError::EmptyInput);
    }
    if !input.len().is_power_of_two() {
        return Err(SciError::InvalidParameter(
            "FFT input length must be a power of two",
        ));
    }

    let n = input.len();
    let mut output = input.to_vec();
    bit_reverse_reorder(&mut output);

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * core::f64::consts::PI / len as f64;
        let w_len = Complex::exp_i(angle);

        for start in (0..n).step_by(len) {
            let mut w = Complex::new(1.0, 0.0);
            for i in 0..half {
                let even = output[start + i];
                let odd = output[start + i + half] * w;
                output[start + i] = even + odd;
                output[start + i + half] = even - odd;
                w = w * w_len;
            }
        }

        len *= 2;
    }

    Ok(output)
}

pub fn ifft(input: &[Complex]) -> SciResult<Vec<Complex>> {
    let conjugated = input
        .iter()
        .map(|value| value.conjugate())
        .collect::<Vec<_>>();
    let transformed = fft(&conjugated)?;
    Ok(transformed
        .into_iter()
        .map(|value| value.conjugate() / input.len() as f64)
        .collect())
}

pub fn power_spectrum(input: &[Complex]) -> SciResult<Vec<f64>> {
    Ok(fft(input)?
        .into_iter()
        .map(|value| value.magnitude().powi(2))
        .collect())
}

fn bit_reverse_reorder(values: &mut [Complex]) {
    let n = values.len();
    let bits = n.trailing_zeros();
    for index in 0..n {
        let reversed = reverse_bits(index, bits);
        if reversed > index {
            values.swap(index, reversed);
        }
    }
}

fn reverse_bits(value: usize, width: u32) -> usize {
    let mut reversed = 0;
    for i in 0..width {
        reversed = (reversed << 1) | ((value >> i) & 1);
    }
    reversed
}
