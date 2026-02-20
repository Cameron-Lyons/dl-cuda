# dl-cuda

A GPU-accelerated deep learning framework written from scratch in CUDA C++.

## Features

**Layers**
- Linear (fully connected)
- Conv1D / Conv2D
- LSTM
- Elman RNN
- Transformer (multi-head attention, layer norm, residual connections)

**Activations** — ReLU, Sigmoid, Tanh

**Loss Functions** — MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy

**Optimizers** — SGD, RMSprop, Adam, AdamW

**Utilities** — Sequential model API

**Examples API** — reusable `run_char_lm(...)` and `run_xor(...)` entry points
via `include/dl_cuda/examples.hpp`

## Requirements

- CMake 3.18+
- NVIDIA CUDA Toolkit
- NVIDIA driver compatible with the CUDA runtime/toolkit version
- C++17 compiler

## Build

```sh
cmake -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

If your GPU architecture is not detected by default, set it explicitly:

```sh
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

## Lint And Warnings

```sh
./scripts/lint.sh
./scripts/format.sh
./scripts/check_no_warnings_build.sh build
```

## Usage

### Character LM

```sh
./build/dl-cuda-char-lm --epochs 800 --print-every 50
# Reuse existing checkpoint and skip saving:
./build/dl-cuda-char-lm --epochs 0 --load-weights --weights model.bin --no-save
```

```
Char-level LM | vocab=..., seq_len=64, d_model=64, d_ff=256, heads=4, layers=3
Optimizer: AdamW (wd=0.01) | Grad clip: 1.0 | Sampling: temp=0.8, top_p=0.9
Training on ... chars of Shakespeare for 800 epochs
...
Weights saved to model.bin
Generating text (temp=0.8, top_p=0.9, 200 chars):
  "To be, or not to be, ..."
```

### XOR

```sh
./build/dl-cuda-xor --epochs 3000 --lr 0.1
```

### Programmatic API

```cpp
#include "dl_cuda/examples.hpp"

int main() {
  dlcuda::CharLMConfig cfg;
  cfg.epochs = 100;
  return dlcuda::run_char_lm(cfg);
}
```

### Profiling

See `docs/PROFILING.md` and run:

```sh
./scripts/profile_char_lm.sh
./scripts/bench_char_lm.sh 200 64
```

## License

MIT
