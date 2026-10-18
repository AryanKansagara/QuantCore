# Local benchmark

Measured October 18, 2026 using Apple Clang 21.0.0, arm64 macOS, `-std=c++17 -O3`. This is one local run, not a general hardware performance claim.

| Measurement | Result |
| --- | ---: |
| Synthetic projection dimensions | 4096 × 4096 |
| Parameters | 16,777,216 |
| FP32 weight bytes | 67,108,864 |
| INT8 weight + scale bytes | 16,793,600 |
| Weight storage reduction | 74.9756% |
| Worst relative L2 error, 8 synthetic inputs | 0.00891761 (0.892%) |
| Mean output MSE | 0.000122874 |
| Minimum output cosine | 0.999960251 |
| FP32 median inference latency | 7.573 ms |
| INT8 weight-only median inference latency | 7.953 ms |
| Sanitized test assertions passed | 2,090 |

Each latency path includes an untimed warm-up and seven timed repetitions. The timed call includes input checks and output allocation; it excludes initial weight generation/quantization and terminal output. The portable INT8 path was slightly slower on this run. Its benefit here is storage reduction, not demonstrated latency improvement. Both paths use simple C++ loops, not optimized BLAS or accelerator kernels.

Reproduce with `make test sanitize benchmark`. Timing and generated samples can vary with hardware, compiler, and standard-library implementation. The executable returns failure if its worst relative L2 error reaches 2%.
