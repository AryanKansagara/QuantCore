# QuantCore: C++ Weight-Only INT8 Inference

A C++17 quantization and inference benchmark for a synthetic 4096 × 4096 linear projection (16,777,216 parameters). It demonstrates per-channel weight-only INT8 quantization, numerical checks, and explicit memory/runtime tradeoffs. Developed with AI assistance and verified with automated tests and sanitizers.

## Quick start

```sh
git clone https://github.com/AryanKansagara/QuantCore.git
cd QuantCore
make test sanitize benchmark
```

Requires a C++17 compiler and Make. The GitHub Actions workflow checks Linux and macOS. See [BENCHMARK.md](BENCHMARK.md) for measured local results, including the fact that the portable INT8 path was slightly slower than FP32 on that run.

## Build

```sh
clang++ -std=c++17 -O3 -Wall -Wextra -Wpedantic benchmark.cpp -o quantcore
./quantcore
clang++ -std=c++17 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined -fno-omit-frame-pointer tests.cpp -o tests
./tests
```

Standard library only. Tested with Apple Clang 21 on arm64 macOS. No accelerator SDK or external ML framework is required for this C++ benchmark.

## Implementation

- Symmetric per-output-channel post-training weight quantization: `scale = max(abs(weights))/127`, rounded and clamped to [-127,127]. Zero rows and very small scales are handled explicitly.
- A weight-only INT8 matvec with FP32 activations/accumulation and per-row output rescaling. No full dequantized weight buffer is materialized.
- Input shape and finite-value validation; RAII-owned contiguous weight and scale buffers.
- Held-out synthetic-vector comparison with the FP32 reference using relative L2 error, MSE, and cosine similarity.
- Warm-up and seven timed repetitions per path, reporting median latency. Tests validate shape errors, nonfinite values, zero rows, channel-specific scales, quantization error bounds, and numerical consistency.

## Interpret results accurately

FP32 weights occupy 67,108,864 bytes. INT8 weights plus FP32 row scales occupy 16,793,600 bytes: **74.9756% less weight storage**, approximately 4× compression. This excludes vector/object overhead, activations, and other process memory; both reference and quantized weights coexist during validation.

The generated weights are synthetic, not learned by training. Evaluation vectors are synthetic and do not measure task accuracy, perplexity, or preservation of LLM capabilities. Floating-point rounding and random-distribution implementations can vary across platforms. The runtime uses portable C++ arithmetic, not native INT8 tensor-core instructions, CUDA kernels, or Tenstorrent kernels. A smaller weight representation does not guarantee faster inference. The FP32 reference is a simple loop, not an optimized BLAS baseline. Timings are local and workload-specific.

No distillation, kernel fusion, real-model calibration, compiler memory planner, or production deployment is implemented. Next steps: load weights from a reviewed PyTorch export, benchmark a trained model on representative data, compare optimized baselines, inspect compiler vectorization, and explore target-specific kernels.
