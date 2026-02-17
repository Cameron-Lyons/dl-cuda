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

- Removed per-epoch one-hot target materialization in Char-LM training.
- Replaced one-hot CE loss/backward with target-id CE kernels.
- Reused `Sequential` forward/backward workspaces to avoid per-step allocations.
- Reduced global `cudaDeviceSynchronize()` calls in the hot training path.

## What to inspect first in Nsight

- Kernel launch counts for CE/loss path.
- Time in transformer attention and feed-forward kernels.
- Memcpy activity between host and device in training loop.
