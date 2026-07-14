# `signal_ext` Module Documentation

Extended signal processing: IIR filter design, 2D FFT, Welch PSD, Hilbert transform.

## Overview

This module extends the core signal tools with filter design, two-dimensional FFT routines, and spectral estimation utilities.

| Feature | Functions |
|---|---|
| IIR design | `butterworth_lowpass`, `butterworth_highpass`, `chebyshev1_lowpass` |
| IIR application | `apply_iir` (direct-form II) |
| 2D FFT | `fft2`, `ifft2`, `fft2_magnitude` |
| Spectral analysis | `welch_psd`, `periodogram` |
| Misc | `hilbert_transform`, `instantaneous_frequency`, `zero_phase_filter` |

## Notes

- Use this module when one-dimensional FFT is not enough.
- Filter design helpers are intended for practical digital-signal workflows.
