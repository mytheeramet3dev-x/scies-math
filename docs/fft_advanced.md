# `fft_advanced` Module Documentation

Advanced FFT utilities and digital filter design.

Builds on the base FFT in `signal.rs`:
- [`fft_real`]           — FFT for real-valued input (returns non-redundant half-spectrum)
- [`stft`]               — Short-Time Fourier Transform with windowing
- [`convolve_fft`]       — Fast O(n log n) linear convolution
- [`correlate_fft`]      — Cross-correlation via FFT
- [`design_fir_lowpass`] — Windowed-sinc FIR low-pass filter coefficients
- [`apply_fir`]          — Apply FIR filter to a signal (causal, zero-initial-state)
- [`spectrogram`]        — Power spectrogram from STFT