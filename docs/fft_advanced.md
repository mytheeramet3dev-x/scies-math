# `fft_advanced` Module Documentation

Advanced FFT utilities and digital filter design.

## Overview

This module builds on the base FFT in `signal.rs` and adds frequency-domain and filter-design helpers for more advanced workflows.

## Topics

- [`fft_real`]           — FFT for real-valued input (returns non-redundant half-spectrum)
- [`stft`]               — Short-Time Fourier Transform with windowing
- [`convolve_fft`]       — Fast O(n log n) linear convolution
- [`correlate_fft`]      — Cross-correlation via FFT
- [`design_fir_lowpass`] — Windowed-sinc FIR low-pass filter coefficients
- [`apply_fir`]          — Apply FIR filter to a signal (causal, zero-initial-state)
- [`spectrogram`]        — Power spectrogram from STFT

## Notes

- Use this module when repeated convolution or spectral analysis becomes a bottleneck.
- The helpers are designed for practical digital-signal workflows rather than symbolic DSP design.
