# `perf` Module Documentation

High-performance matrix multiplication — Phase 1 optimisations.

Three backends, selected automatically by matrix size:

| Size (max dim) | Backend |
|---|---|
| < 64 | Naive (fast for tiny matrices) |
| 64 – 511 | Cache-tiled (TILE × TILE blocks) |
| ≥ 512 | Strassen recursive + tiling at leaves |

All backends are pure-Rust, zero-dependency.
An optional multithreaded backend (`matmul_threaded`) is available
for square matrices where n ≥ 256.