//! Advanced FFT utilities and digital filter design.
//!
//! Builds on the base FFT in `signal.rs`:
//! - [`fft_real`]           — FFT for real-valued input (returns non-redundant half-spectrum)
//! - [`stft`]               — Short-Time Fourier Transform with windowing
//! - [`convolve_fft`]       — Fast O(n log n) linear convolution
//! - [`correlate_fft`]      — Cross-correlation via FFT
//! - [`design_fir_lowpass`] — Windowed-sinc FIR low-pass filter coefficients
//! - [`apply_fir`]          — Apply FIR filter to a signal (causal, zero-initial-state)
//! - [`spectrogram`]        — Power spectrogram from STFT

use crate::errors::{SciError, SciResult};
use crate::signal::{Complex, fft, ifft};

// ── Window functions ──────────────────────────────────────────────────────────

/// Rectangular (Dirichlet) window.
pub fn window_rect(size: usize) -> Vec<f64> {
    vec![1.0; size]
}

/// Hann window.
pub fn window_hann(size: usize) -> Vec<f64> {
    (0..size)
        .map(|n| 0.5 * (1.0 - (2.0 * core::f64::consts::PI * n as f64 / (size - 1) as f64).cos()))
        .collect()
}

/// Hamming window.
pub fn window_hamming(size: usize) -> Vec<f64> {
    (0..size)
        .map(|n| 0.54 - 0.46 * (2.0 * core::f64::consts::PI * n as f64 / (size - 1) as f64).cos())
        .collect()
}

/// Blackman window.
pub fn window_blackman(size: usize) -> Vec<f64> {
    let pi = core::f64::consts::PI;
    (0..size)
        .map(|n| {
            let t = 2.0 * pi * n as f64 / (size - 1) as f64;
            0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
        })
        .collect()
}

// ── Real-signal FFT ───────────────────────────────────────────────────────────

/// FFT specialised for a **real** signal.
///
/// Returns the first `n/2 + 1` complex bins (the rest are conjugate mirrors).
pub fn fft_real(signal: &[f64]) -> SciResult<Vec<Complex>> {
    if signal.is_empty() {
        return Err(SciError::EmptyInput);
    }
    if !signal.len().is_power_of_two() {
        return Err(SciError::InvalidParameter(
            "fft_real input length must be a power of two",
        ));
    }
    let input: Vec<Complex> = signal.iter().map(|&r| Complex::new(r, 0.0)).collect();
    let full = fft(&input)?;
    Ok(full[..=signal.len() / 2].to_vec())
}

// ── Short-Time Fourier Transform ──────────────────────────────────────────────

/// Short-Time Fourier Transform (STFT).
///
/// Returns a 2-D array of complex spectra: `frames × (frame_size/2 + 1)`.
///
/// # Parameters
/// - `signal`     — input real signal
/// - `frame_size` — FFT size (must be power of two)
/// - `hop`        — number of samples between successive frames
/// - `window`     — window coefficients (length must equal `frame_size`)
pub fn stft(
    signal: &[f64],
    frame_size: usize,
    hop: usize,
    window: &[f64],
) -> SciResult<Vec<Vec<Complex>>> {
    if signal.is_empty() {
        return Err(SciError::EmptyInput);
    }
    if !frame_size.is_power_of_two() {
        return Err(SciError::InvalidParameter(
            "frame_size must be a power of two",
        ));
    }
    if hop == 0 {
        return Err(SciError::InvalidParameter("hop must be positive"));
    }
    if window.len() != frame_size {
        return Err(SciError::InvalidParameter(
            "window length must equal frame_size",
        ));
    }

    let mut frames = Vec::new();
    let mut start = 0usize;
    while start + frame_size <= signal.len() {
        let frame: Vec<Complex> = (0..frame_size)
            .map(|i| Complex::new(signal[start + i] * window[i], 0.0))
            .collect();
        let spectrum = fft(&frame)?;
        frames.push(spectrum[..=frame_size / 2].to_vec());
        start += hop;
    }
    if frames.is_empty() {
        return Err(SciError::InvalidParameter(
            "signal is too short for the given frame_size",
        ));
    }
    Ok(frames)
}

// ── Convolution / Correlation ─────────────────────────────────────────────────

/// Fast linear convolution via FFT: `out = a ⊛ b`.
///
/// Output length = `a.len() + b.len() - 1`.
pub fn convolve_fft(a: &[f64], b: &[f64]) -> SciResult<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Err(SciError::EmptyInput);
    }
    let out_len = a.len() + b.len() - 1;
    let fft_len = next_power_of_two(out_len);

    let fa = fft_padded(a, fft_len)?;
    let fb = fft_padded(b, fft_len)?;

    let product: Vec<Complex> = fa.iter().zip(fb.iter()).map(|(&x, &y)| x * y).collect();
    let result = ifft(&product)?;
    Ok(result[..out_len].iter().map(|c| c.re).collect())
}

/// Fast cross-correlation: `out[k] = sum_n a[n] * b[n+k]`.
///
/// Output length = `a.len() + b.len() - 1`.
pub fn correlate_fft(a: &[f64], b: &[f64]) -> SciResult<Vec<f64>> {
    if a.is_empty() || b.is_empty() {
        return Err(SciError::EmptyInput);
    }
    let out_len = a.len() + b.len() - 1;
    let fft_len = next_power_of_two(out_len);

    let fa = fft_padded(a, fft_len)?;
    let fb = fft_padded(b, fft_len)?;

    // Correlation = conj(FA) * FB in frequency domain.
    let product: Vec<Complex> = fa
        .iter()
        .zip(fb.iter())
        .map(|(&x, &y)| x.conjugate() * y)
        .collect();
    let result = ifft(&product)?;
    // Rearrange: lags from -(b.len()-1) to (a.len()-1).
    let neg = b.len() - 1;
    let mut out = vec![0.0f64; out_len];
    for k in 0..out_len {
        let idx = (k + fft_len - neg) % fft_len;
        out[k] = result[idx].re;
    }
    Ok(out)
}

// ── FIR Filter Design ─────────────────────────────────────────────────────────

/// Design a windowed-sinc **FIR low-pass** filter.
///
/// # Parameters
/// - `cutoff_norm` — normalised cutoff frequency in (0, 0.5) where 0.5 = Nyquist
/// - `num_taps`    — filter length (odd recommended for linear phase)
/// - `window`      — window coefficients of length `num_taps`
pub fn design_fir_lowpass(
    cutoff_norm: f64,
    num_taps: usize,
    window: &[f64],
) -> SciResult<Vec<f64>> {
    if !(0.0..0.5).contains(&cutoff_norm) {
        return Err(SciError::DomainError("cutoff_norm must be in (0, 0.5)"));
    }
    if num_taps == 0 || window.len() != num_taps {
        return Err(SciError::InvalidParameter(
            "num_taps must be positive and window length must match",
        ));
    }

    let m = (num_taps - 1) as f64 / 2.0;
    let pi = core::f64::consts::PI;
    let coeffs: Vec<f64> = (0..num_taps)
        .map(|n| {
            let k = n as f64 - m;
            let h = if k.abs() < f64::EPSILON {
                2.0 * cutoff_norm
            } else {
                (2.0 * pi * cutoff_norm * k).sin() / (pi * k)
            };
            h * window[n]
        })
        .collect();

    // Normalise for unity DC gain.
    let gain: f64 = coeffs.iter().sum();
    if gain.abs() < f64::EPSILON {
        return Err(SciError::DivisionByZero);
    }
    Ok(coeffs.iter().map(|&c| c / gain).collect())
}

/// Design a windowed-sinc **FIR high-pass** filter (spectral inversion of low-pass).
pub fn design_fir_highpass(
    cutoff_norm: f64,
    num_taps: usize,
    window: &[f64],
) -> SciResult<Vec<f64>> {
    let lp = design_fir_lowpass(cutoff_norm, num_taps, window)?;
    let mid = num_taps / 2;
    Ok(lp
        .iter()
        .enumerate()
        .map(|(i, &c)| if i == mid { 1.0 - c } else { -c })
        .collect())
}

/// Apply a FIR filter to a signal (direct-form, causal, zero-padded).
pub fn apply_fir(signal: &[f64], coeffs: &[f64]) -> SciResult<Vec<f64>> {
    if signal.is_empty() {
        return Err(SciError::EmptyInput);
    }
    if coeffs.is_empty() {
        return Err(SciError::InvalidParameter(
            "filter coefficients must be non-empty",
        ));
    }
    let h = coeffs.len();
    Ok((0..signal.len())
        .map(|n| {
            (0..h)
                .filter(|&k| k <= n)
                .map(|k| coeffs[k] * signal[n - k])
                .sum()
        })
        .collect())
}

// ── Power Spectrogram ─────────────────────────────────────────────────────────

/// Compute the power spectrogram (magnitude² per bin) from STFT frames.
pub fn spectrogram(stft_frames: &[Vec<Complex>]) -> Vec<Vec<f64>> {
    stft_frames
        .iter()
        .map(|frame| frame.iter().map(|c| c.magnitude().powi(2)).collect())
        .collect()
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn next_power_of_two(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

fn fft_padded(signal: &[f64], len: usize) -> SciResult<Vec<Complex>> {
    let mut input: Vec<Complex> = signal.iter().map(|&r| Complex::new(r, 0.0)).collect();
    input.resize(len, Complex::zero());
    fft(&input)
}
