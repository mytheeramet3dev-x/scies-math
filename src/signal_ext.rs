//! Extended signal processing: IIR filter design, 2D FFT, Welch PSD, Hilbert transform.
//!
//! | Feature | Functions |
//! |---|---|
//! | IIR design | `butterworth_lowpass`, `butterworth_highpass`, `chebyshev1_lowpass` |
//! | IIR application | `apply_iir` (direct-form II) |
//! | 2D FFT | `fft2`, `ifft2`, `fft2_magnitude` |
//! | Spectral analysis | `welch_psd`, `periodogram` |
//! | Misc | `hilbert_transform`, `instantaneous_frequency`, `zero_phase_filter` |

use crate::errors::{SciError, SciResult};
use crate::signal::{Complex, fft, ifft};

// ══════════════════════════════════════════════════════════════════════════════
// IIR filter design (analog prototype → bilinear transform → digital)
// ══════════════════════════════════════════════════════════════════════════════

/// Coefficients of a digital IIR filter (Direct Form II).
///
/// Transfer function: H(z) = B(z)/A(z)
/// where `b` = numerator, `a` = denominator (`a[0]` normalised to 1).
#[derive(Debug, Clone)]
pub struct IirFilter {
    pub b: Vec<f64>,
    pub a: Vec<f64>,
}

impl IirFilter {
    /// Order of the filter.
    pub fn order(&self) -> usize {
        self.a.len() - 1
    }
}

/// Design an N-th order **Butterworth low-pass** digital filter.
///
/// `cutoff_norm` ∈ (0, 1): normalised cutoff frequency (1.0 = Nyquist).
/// Uses the bilinear transform from an analog Butterworth prototype.
pub fn butterworth_lowpass(order: usize, cutoff_norm: f64) -> SciResult<IirFilter> {
    if order == 0 || order > 10 {
        return Err(SciError::InvalidParameter("order must be in [1, 10]"));
    }
    if !(0.0..1.0).contains(&cutoff_norm) {
        return Err(SciError::DomainError("cutoff_norm must be in (0, 1)"));
    }
    let pi = core::f64::consts::PI;
    // Pre-warp: Ω_c = 2·tan(π·fc)
    let omega_c = 2.0 * (pi * cutoff_norm / 2.0).tan();
    // Analog Butterworth poles: s_k = Ω_c · exp(jπ(2k+N-1)/(2N))
    let poles: Vec<(f64, f64)> = (0..order)
        .map(|k| {
            let theta = pi * (2 * k + order - 1) as f64 / (2 * order) as f64;
            (omega_c * theta.cos(), omega_c * theta.sin())
        })
        .collect();
    // Bilinear transform each pole: z = (1 + s/2)/(1 - s/2)
    let digital_poles: Vec<(f64, f64)> = poles
        .iter()
        .map(|&(sr, si)| {
            let nr = 1.0 + sr / 2.0;
            let ni = si / 2.0;
            let dr = 1.0 - sr / 2.0;
            let di = -si / 2.0;
            let denom = dr * dr + di * di;
            ((nr * dr + ni * di) / denom, (ni * dr - nr * di) / denom)
        })
        .collect();
    // All digital zeros at z = -1 (lowpass prototype zeros at s = ∞)
    let digital_zeros: Vec<(f64, f64)> = vec![(-1.0, 0.0); order];
    // Build polynomial coefficients from roots
    let b = roots_to_poly(&digital_zeros);
    let a = roots_to_poly(&digital_poles);
    // Normalise gain at DC (z=1): H(1) = 1
    let gain_b: f64 = b.iter().sum();
    let gain_a: f64 = a.iter().sum();
    let k = gain_a / gain_b;
    Ok(IirFilter {
        b: b.iter().map(|&v| v * k).collect(),
        a,
    })
}

/// Design an N-th order **Butterworth high-pass** digital filter.
pub fn butterworth_highpass(order: usize, cutoff_norm: f64) -> SciResult<IirFilter> {
    if order == 0 || order > 10 {
        return Err(SciError::InvalidParameter("order must be in [1, 10]"));
    }
    if !(0.0..1.0).contains(&cutoff_norm) {
        return Err(SciError::DomainError("cutoff_norm must be in (0, 1)"));
    }
    let pi = core::f64::consts::PI;
    let omega_c = 2.0 * (pi * cutoff_norm / 2.0).tan();
    // High-pass analog prototype: replace s → Ω_c/s
    let poles: Vec<(f64, f64)> = (0..order)
        .map(|k| {
            let theta = pi * (2 * k + order - 1) as f64 / (2 * order) as f64;
            let sr = omega_c * theta.cos();
            let si = omega_c * theta.sin();
            // s_hp = Ω_c / s_lp
            let denom = sr * sr + si * si;
            (omega_c * sr / denom, -omega_c * si / denom)
        })
        .collect();
    let digital_poles: Vec<(f64, f64)> = poles
        .iter()
        .map(|&(sr, si)| {
            let nr = 1.0 + sr / 2.0;
            let ni = si / 2.0;
            let dr = 1.0 - sr / 2.0;
            let di = -si / 2.0;
            let den = dr * dr + di * di;
            ((nr * dr + ni * di) / den, (ni * dr - nr * di) / den)
        })
        .collect();
    // High-pass zeros at z = +1
    let digital_zeros: Vec<(f64, f64)> = vec![(1.0, 0.0); order];
    let b = roots_to_poly(&digital_zeros);
    let a = roots_to_poly(&digital_poles);
    // Normalise gain at Nyquist (z=-1)
    let gain_b: f64 = b
        .iter()
        .enumerate()
        .map(|(i, &v)| v * if i % 2 == 0 { 1.0 } else { -1.0 })
        .sum();
    let gain_a: f64 = a
        .iter()
        .enumerate()
        .map(|(i, &v)| v * if i % 2 == 0 { 1.0 } else { -1.0 })
        .sum();
    let k = gain_a / gain_b;
    Ok(IirFilter {
        b: b.iter().map(|&v| v * k).collect(),
        a,
    })
}

/// Design an N-th order **Chebyshev Type-I low-pass** digital filter.
///
/// `ripple_db` — passband ripple in dB (e.g. 0.5 dB).
pub fn chebyshev1_lowpass(order: usize, cutoff_norm: f64, ripple_db: f64) -> SciResult<IirFilter> {
    if order == 0 || order > 8 {
        return Err(SciError::InvalidParameter("order must be in [1, 8]"));
    }
    if !(0.0..1.0).contains(&cutoff_norm) {
        return Err(SciError::DomainError("cutoff_norm must be in (0, 1)"));
    }
    if ripple_db <= 0.0 {
        return Err(SciError::InvalidParameter("ripple_db must be positive"));
    }
    let pi = core::f64::consts::PI;
    let omega_c = 2.0 * (pi * cutoff_norm / 2.0).tan();
    let eps = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();
    let n = order as f64;
    let sigma_0 = (1.0 / eps).asinh() / n;
    // Analog Chebyshev I poles
    let poles: Vec<(f64, f64)> = (1..=(order))
        .map(|k| {
            let theta = pi * (2 * k - 1) as f64 / (2.0 * n);
            let sr = -omega_c * sigma_0.sinh() * theta.sin();
            let si = omega_c * sigma_0.cosh() * theta.cos();
            (sr, si)
        })
        .collect();
    let digital_poles: Vec<(f64, f64)> = poles
        .iter()
        .map(|&(sr, si)| {
            let nr = 1.0 + sr / 2.0;
            let ni = si / 2.0;
            let dr = 1.0 - sr / 2.0;
            let di = -si / 2.0;
            let den = dr * dr + di * di;
            ((nr * dr + ni * di) / den, (ni * dr - nr * di) / den)
        })
        .collect();
    let digital_zeros: Vec<(f64, f64)> = vec![(-1.0, 0.0); order];
    let b = roots_to_poly(&digital_zeros);
    let a = roots_to_poly(&digital_poles);
    let gain_b: f64 = b.iter().sum();
    let gain_a: f64 = a.iter().sum();
    let k = gain_a / gain_b;
    Ok(IirFilter {
        b: b.iter().map(|&v| v * k).collect(),
        a,
    })
}

/// Design a **notch filter** at normalised frequency `freq_norm` with quality factor `q`.
pub fn notch_filter(freq_norm: f64, q: f64) -> SciResult<IirFilter> {
    if !(0.0..1.0).contains(&freq_norm) {
        return Err(SciError::DomainError("freq_norm ∈ (0,1)"));
    }
    if q <= 0.0 {
        return Err(SciError::InvalidParameter("q must be positive"));
    }
    let pi = core::f64::consts::PI;
    let w0 = pi * freq_norm;
    let bw = w0 / q;
    let r = 1.0 - bw / 2.0;
    let k = (1.0 - 2.0 * r * w0.cos() + r * r) / (2.0 - 2.0 * w0.cos());
    Ok(IirFilter {
        b: vec![k, -2.0 * k * w0.cos(), k],
        a: vec![1.0, -2.0 * r * w0.cos(), r * r],
    })
}

/// Apply an IIR filter (Direct Form II) to a signal.
///
/// Returns the filtered signal.
pub fn apply_iir(signal: &[f64], filter: &IirFilter) -> SciResult<Vec<f64>> {
    let nb = filter.b.len();
    let na = filter.a.len();
    if nb == 0 || na == 0 {
        return Err(SciError::InvalidParameter("empty filter"));
    }
    let n = signal.len();
    let m = nb.max(na);
    let mut w = vec![0.0f64; m]; // delay line
    let mut out = Vec::with_capacity(n);
    let a0 = filter.a[0];
    for &x in signal {
        // Direct Form II: w[0] = x - sum(a[k]*w[k]) / a[0]
        let mut wn = x;
        for k in 1..na {
            wn -= filter.a[k] / a0 * w[k - 1];
        }
        let y: f64 = (0..nb)
            .map(|k| filter.b[k] / a0 * if k == 0 { wn } else { w[k - 1] })
            .sum();
        // Shift delay line
        for k in (1..m).rev() {
            w[k] = w[k - 1];
        }
        w[0] = wn;
        out.push(y);
    }
    Ok(out)
}

/// Zero-phase filtering (forward + backward pass) to eliminate phase distortion.
pub fn zero_phase_filter(signal: &[f64], filter: &IirFilter) -> SciResult<Vec<f64>> {
    let fwd = apply_iir(signal, filter)?;
    let rev: Vec<f64> = fwd.iter().rev().copied().collect();
    let bwd = apply_iir(&rev, filter)?;
    Ok(bwd.iter().rev().copied().collect())
}

// ══════════════════════════════════════════════════════════════════════════════
// 2D FFT
// ══════════════════════════════════════════════════════════════════════════════

/// 2D FFT of a real or complex image stored in **row-major** format.
///
/// `rows × cols` complex values → 2D DFT coefficients (same layout).
pub fn fft2(data: &[Complex], rows: usize, cols: usize) -> SciResult<Vec<Complex>> {
    if data.len() != rows * cols {
        return Err(SciError::InvalidParameter("data length != rows*cols"));
    }
    let mut out = data.to_vec();
    // Row-wise FFT
    for r in 0..rows {
        let row: Vec<Complex> = out[r * cols..(r + 1) * cols].to_vec();
        let row_fft = fft(&row)?;
        out[r * cols..(r + 1) * cols].copy_from_slice(&row_fft);
    }
    // Column-wise FFT
    for c in 0..cols {
        let col: Vec<Complex> = (0..rows).map(|r| out[r * cols + c]).collect();
        let col_fft = fft(&col)?;
        for r in 0..rows {
            out[r * cols + c] = col_fft[r];
        }
    }
    Ok(out)
}

/// 2D inverse FFT (unnormalised; divides by rows*cols).
pub fn ifft2(data: &[Complex], rows: usize, cols: usize) -> SciResult<Vec<Complex>> {
    if data.len() != rows * cols {
        return Err(SciError::InvalidParameter("data length != rows*cols"));
    }
    // Conjugate → FFT2 → conjugate → divide by N
    let conj: Vec<Complex> = data.iter().map(|c| c.conjugate()).collect();
    let fwd = fft2(&conj, rows, cols)?;
    let n = (rows * cols) as f64;
    Ok(fwd
        .iter()
        .map(|c| Complex {
            re: c.conjugate().re / n,
            im: c.conjugate().im / n,
        })
        .collect())
}

/// 2D FFT magnitude spectrum (useful for display/analysis).
pub fn fft2_magnitude(data: &[f64], rows: usize, cols: usize) -> SciResult<Vec<f64>> {
    let complex: Vec<Complex> = data.iter().map(|&v| Complex { re: v, im: 0.0 }).collect();
    let spectrum = fft2(&complex, rows, cols)?;
    Ok(spectrum.iter().map(|c| c.magnitude()).collect())
}

// ══════════════════════════════════════════════════════════════════════════════
// Power Spectral Density — Welch's method
// ══════════════════════════════════════════════════════════════════════════════

/// **Welch's power spectral density** estimate.
///
/// Splits `signal` into overlapping segments, windows each, FFTs, and averages.
///
/// Returns `(frequencies, psd)` where `frequencies[k] = k / (segment_len * sample_rate)`.
///
/// # Parameters
/// - `segment_len` — FFT size per segment (power of 2 recommended)
/// - `overlap` — number of overlapping samples between segments  
/// - `sample_rate` — Hz (used only to compute frequency axis)
pub fn welch_psd(
    signal: &[f64],
    segment_len: usize,
    overlap: usize,
    sample_rate: f64,
) -> SciResult<(Vec<f64>, Vec<f64>)> {
    if segment_len < 4 {
        return Err(SciError::InvalidParameter("segment_len >= 4"));
    }
    if overlap >= segment_len {
        return Err(SciError::InvalidParameter("overlap < segment_len"));
    }
    let step = segment_len - overlap;
    let n_freqs = segment_len / 2 + 1;
    let window: Vec<f64> = (0..segment_len)
        .map(|i| {
            let pi = core::f64::consts::PI;
            0.5 - 0.5 * (2.0 * pi * i as f64 / (segment_len - 1) as f64).cos() // Hann
        })
        .collect();
    let win_power: f64 = window.iter().map(|w| w * w).sum::<f64>() / segment_len as f64;

    let mut psd = vec![0.0f64; n_freqs];
    let mut n_segments = 0usize;
    let mut start = 0usize;
    while start + segment_len <= signal.len() {
        let seg: Vec<Complex> = (0..segment_len)
            .map(|i| Complex {
                re: signal[start + i] * window[i],
                im: 0.0,
            })
            .collect();
        let spec = fft(&seg)?;
        for k in 0..n_freqs {
            let power = spec[k].re * spec[k].re + spec[k].im * spec[k].im;
            psd[k] += power;
        }
        n_segments += 1;
        start += step;
    }
    if n_segments == 0 {
        return Err(SciError::InvalidParameter("signal too short for segment"));
    }
    let scale = 2.0 / (sample_rate * segment_len as f64 * win_power * n_segments as f64);
    let psd_scaled: Vec<f64> = psd
        .iter()
        .enumerate()
        .map(|(k, &p)| p * scale * if k == 0 || k == n_freqs - 1 { 0.5 } else { 1.0 })
        .collect();
    let freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * sample_rate / segment_len as f64)
        .collect();
    Ok((freqs, psd_scaled))
}

/// Simple periodogram (squared magnitude of FFT, normalised by N).
pub fn periodogram(signal: &[f64], sample_rate: f64) -> SciResult<(Vec<f64>, Vec<f64>)> {
    let n = signal.len();
    let spec = fft(&signal
        .iter()
        .map(|&v| Complex { re: v, im: 0.0 })
        .collect::<Vec<_>>())?;
    let n_freqs = n / 2 + 1;
    let psd: Vec<f64> = (0..n_freqs)
        .map(|k| (spec[k].re * spec[k].re + spec[k].im * spec[k].im) / (n as f64 * sample_rate))
        .collect();
    let freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * sample_rate / n as f64)
        .collect();
    Ok((freqs, psd))
}

// ══════════════════════════════════════════════════════════════════════════════
// Hilbert transform
// ══════════════════════════════════════════════════════════════════════════════

/// **Hilbert transform** of a real signal via FFT.
///
/// Returns the analytic signal (complex): `z[n] = x[n] + j·H{x}[n]`.
/// The imaginary part is the Hilbert transform.
pub fn hilbert(signal: &[f64]) -> SciResult<Vec<Complex>> {
    let n = signal.len();
    if n < 2 {
        return Err(SciError::InvalidParameter("signal must have >= 2 samples"));
    }
    let spec = fft(&signal
        .iter()
        .map(|&v| Complex { re: v, im: 0.0 })
        .collect::<Vec<_>>())?;
    // Build analytic signal: double positive freqs, zero negative freqs
    let mut h = vec![Complex { re: 0.0, im: 0.0 }; n];
    h[0] = spec[0]; // DC
    if n % 2 == 0 {
        h[n / 2] = spec[n / 2];
    } // Nyquist
    for k in 1..(n / 2) {
        h[k] = Complex {
            re: spec[k].re * 2.0,
            im: spec[k].im * 2.0,
        };
    }
    // IFFT
    ifft(&h)
}

/// **Instantaneous frequency** from the analytic signal's phase derivative.
///
/// `sample_rate` in Hz; returns instantaneous frequency in Hz for each sample.
pub fn instantaneous_frequency(analytic: &[Complex], sample_rate: f64) -> Vec<f64> {
    let n = analytic.len();
    let mut ifreq = vec![0.0f64; n];
    for i in 1..n {
        let dphase =
            (analytic[i].im).atan2(analytic[i].re) - (analytic[i - 1].im).atan2(analytic[i - 1].re);
        // Unwrap phase difference to (-π, π)
        let dp = dphase.rem_euclid(2.0 * core::f64::consts::PI);
        let dp = if dp > core::f64::consts::PI {
            dp - 2.0 * core::f64::consts::PI
        } else {
            dp
        };
        ifreq[i] = dp * sample_rate / (2.0 * core::f64::consts::PI);
    }
    ifreq[0] = ifreq[1];
    ifreq
}

/// **Envelope** (instantaneous amplitude) of the analytic signal.
pub fn envelope(analytic: &[Complex]) -> Vec<f64> {
    analytic.iter().map(|c| c.magnitude()).collect()
}

// ══════════════════════════════════════════════════════════════════════════════
// Window functions (additional)
// ══════════════════════════════════════════════════════════════════════════════

/// Flat-top window — optimal for amplitude accuracy.
pub fn window_flattop(n: usize) -> Vec<f64> {
    let pi = core::f64::consts::PI;
    (0..n)
        .map(|i| {
            let x = 2.0 * pi * i as f64 / (n - 1) as f64;
            0.21557895 - 0.41663158 * x.cos() + 0.27726316 * (2.0 * x).cos()
                - 0.08357895 * (3.0 * x).cos()
                + 0.00694737 * (4.0 * x).cos()
        })
        .collect()
}

/// Kaiser window with shape parameter `beta`.
pub fn window_kaiser(n: usize, beta: f64) -> Vec<f64> {
    let i0_beta = bessel_i0(beta);
    (0..n)
        .map(|i| {
            let x = 2.0 * i as f64 / (n - 1) as f64 - 1.0; // ∈ [-1, 1]
            bessel_i0(beta * (1.0 - x * x).sqrt()) / i0_beta
        })
        .collect()
}

// ─── Private helpers ──────────────────────────────────────────────────────────

/// Convert complex roots to real polynomial coefficients via polynomial multiplication.
/// For each conjugate pair we get a quadratic with real coefficients.
fn roots_to_poly(roots: &[(f64, f64)]) -> Vec<f64> {
    let mut poly = vec![1.0f64]; // p(z) = 1
    for &(re, im) in roots {
        if im.abs() < 1e-12 {
            // Real root: (z - re)
            let mut new_poly = vec![0.0f64; poly.len() + 1];
            for (i, &c) in poly.iter().enumerate() {
                new_poly[i] += c;
                new_poly[i + 1] -= c * re;
            }
            poly = new_poly;
        } else {
            // Complex conjugate pair: (z - re - j·im)(z - re + j·im) = z² - 2re·z + |root|²
            let b = -2.0 * re;
            let c = re * re + im * im;
            let mut new_poly = vec![0.0f64; poly.len() + 2];
            for (i, &p) in poly.iter().enumerate() {
                new_poly[i] += p;
                new_poly[i + 1] += p * b;
                new_poly[i + 2] += p * c;
            }
            poly = new_poly;
        }
    }
    poly
}

/// Modified Bessel function I₀(x) via series (used in Kaiser window).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let x2 = (x / 2.0) * (x / 2.0);
    for k in 1..30 {
        term *= x2 / (k * k) as f64;
        sum += term;
        if term < sum * 1e-12 {
            break;
        }
    }
    sum
}
