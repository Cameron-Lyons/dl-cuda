# dl-cuda

A GPU-accelerated deep learning framework in CUDA C++.

## v2 (Backward-Incompatible) Architecture

This repository now uses a v2 API with breaking changes:

- `Tensor` + typed shapes/dtypes (`float32`, `int32`) instead of raw pointer I/O
- `RuntimeContext` for cuBLAS/TF32/seed/stream control (no global runtime state)
- `Module`/`Sequential` with explicit ownership (`std::unique_ptr`)
- Explicit `Status`/`Result<T>` error propagation (no `exit(...)` fast-fail path)
- Named state-dict checkpoints with metadata + strict validation
- Public SDK exposed under `include/dl_cuda/*`
- Unified CLI with subcommands (`train-xor`, `train-char`, `sample-char`)

## Build

```sh
cmake -S . -B build
cmake --build build -j
```

If your GPU architecture is not detected by default, set it explicitly:

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

## CLI

```sh
./build/dl-cuda train-xor --epochs 3000 --lr 0.1
./build/dl-cuda train-char --epochs 800 --print-every 50
./build/dl-cuda sample-char --checkpoint char_v2.ckpt --gen-len 200
```

Use config files (key=value) with any subcommand:

```sh
./build/dl-cuda train-char --config configs/char_train.cfg
```

## Programmatic API

```cpp
#include "dl_cuda.hpp"

int main() {
  dlcuda::TrainXorConfig cfg;
  cfg.epochs = 1000;
  dlcuda::Status status = dlcuda::TrainXor(cfg);
  return status.ok() ? 0 : 1;
}
```

## Checkpoints

v2 checkpoints store:

- format/version metadata
- model name
- named tensors (name, dtype, shape, raw bytes)

`LoadCheckpoint(...)` validates model name + tensor schema before loading.

## License

MIT
