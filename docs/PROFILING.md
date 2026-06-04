# Profiling

This project includes a lightweight profiling script for the Char-LM example.

## Prerequisites

- NVIDIA driver + CUDA toolkit runtime compatibility
- Nsight Systems (`nsys`) installed and on `PATH`

## Quick Start

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./scripts/profile_char_lm.sh
```

The script writes an `.nsys-rep` report under `profiles/` and prints a CLI summary.

## Current hotspot improvements already applied

- Trains categorical cross-entropy directly from logits, avoiding training-time softmax.
- Uses block-per-row softmax/logits CE kernels for row reductions.
- Pre-encodes the corpus once and fills training windows on device.
- Updates generation context on device instead of copying the full context each step.
- Reuses tensors and runtime scratch buffers across training steps.
- Skips first-layer input-gradient work when callers only need parameter gradients.
- Uses cuBLAS for linear layers with TF32 enabled by default where available.

## What to inspect first in Nsight

- Kernel launch counts for logits CE, optimizer, and linear layers.
- cuBLAS GEMM time versus custom-kernel fallback time when `--no-cublas` is used.
- Memcpy activity during generation and metric logging.
