# Digital Signal Processing (`signal`, `signal_ext`)

The `signal` and `signal_ext` modules provide Cooley-Tukey Radix-2 Fast Fourier Transforms (1D and 2D), linear convolution, digital filter synthesis (Butterworth, Chebyshev Type I), zero-phase bidirectional filtering, Welch Power Spectral Density estimation, and the Hilbert transform.

---

## 1. Fourier Transforms and Spectral Analysis

### Radix-2 Cooley-Tukey FFT (`fft`, `ifft`)
Computes the Discrete Fourier Transform (DFT) of a sequence of complex numbers $x_0, \dots, x_{N-1}$:

$$X_k = \sum_{n=0}^{N-1} x_n \exp\left(-i \frac{2\pi k n}{N}\right), \quad k = 0, \dots, N-1$$

Transforms run in $O(N \log N)$ operations for lengths $N = 2^m$.

### 2D Fast Fourier Transform (`fft2`, `ifft2`)
Performs separable row-column 2D DFT on $M \times N$ matrices for 2D image processing and spatial frequency decomposition.

### Welch's Power Spectral Density (`welch_psd`)
Estimates the power spectral density $S_{xx}(f)$ by dividing the signal into overlapping segments, applying a Hann window, computing periodograms, and averaging across windows to reduce spectral variance.

---

## 2. Digital Filter Synthesis (`IirFilter`)

Designs analog filter prototypes and converts them to discrete $Z$-domain transfer functions via the **Bilinear Transform with frequency pre-warping**:

$$s = \frac{2}{\Delta t} \frac{1 - z^{-1}}{1 + z^{-1}}, \quad \Omega_c = \frac{2}{\Delta t} \tan\left(\frac{\omega_c \Delta t}{2}\right)$$

### Available Filter Synthesizers
- `butterworth_lowpass(order, cutoff_norm)`: Maximally flat passband, zero ripple.
- `butterworth_highpass(order, cutoff_norm)`: Highpass equivalent.
- `chebyshev1_lowpass(order, cutoff_norm, ripple_db)`: Steeper transition roll-off with controlled passband ripple.
- `zero_phase_filter(signal, filter)`: Forward-backward filtering ($\text{filtfilt}$) ensuring exactly **zero phase distortion**.

---

## 3. Analytic Signal and Hilbert Transform (`hilbert_transform`)

Forms the complex analytic signal $z(t) = x(t) + i \mathcal{H}[x(t)]$:

$$\mathcal{H}[x](t) = \frac{1}{\pi} \text{p.v.} \int_{-\infty}^\infty \frac{x(\tau)}{t - \tau} d\tau$$

Extracts the **instantaneous amplitude (envelope)** $A(t) = |z(t)|$ and **instantaneous frequency** $\omega(t) = \frac{d}{dt} \arg(z(t))$.

---

## 4. Code Example

```rust
use scies_math_th::signal::{fft, Complex};
use scies_math_th::signal_ext::{apply_iir, butterworth_lowpass};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Synthesize a 4th-order Butterworth low-pass filter
    let filter = butterworth_lowpass(4, 0.2)?; // Cutoff at 0.2 of Nyquist
    println!("Filter order: {}, b-coeffs: {}", filter.order(), filter.b.len());

    // 2. Filter a noisy input signal
    let signal = vec![1.0, 0.5, -0.2, 0.8, -0.4, 0.1, 0.9, -0.7];
    let filtered = apply_iir(&signal, &filter)?;
    println!("Filtered signal length: {}", filtered.len());

    // 3. 8-point FFT
    let c_signal: Vec<Complex> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    let spectrum = fft(&c_signal)?;
    println!("DC component X[0]: {:.4} + {:.4}i", spectrum[0].re, spectrum[0].im);

    Ok(())
}
```
