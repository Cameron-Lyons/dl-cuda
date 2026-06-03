# dl-cuda

A GPU-accelerated deep learning framework in CUDA C++.

## v2 (Backward-Incompatible) Architecture

This repository now uses a v2 API with breaking changes:

- `Tensor` + typed shapes/dtypes (`float32`, `int32`) instead of raw pointer I/O
- `RuntimeContext` for cuBLAS/TF32/seed/stream control (no global runtime state)
- `Module`/`Sequential` with explicit ownership (`std::unique_ptr`) and stable parameter caches
- Explicit `Status`/`Result<T>` error propagation (no `exit(...)` fast-fail path)
- Named state-dict checkpoints with metadata + strict validation
- Public core SDK exposed under `include/dl_cuda/*`
- Example workflows split out of the core target and header aggregate
- Unified CLI with declarative option/config parsing (`train-xor`, `train-char`, `sample-char`)

## Build

```sh
cmake -S . -B build
cmake --build build -j
```

If your GPU architecture is not detected by default, set it explicitly:

```sh
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

If you only want host-side tests and parser checks on a machine without `nvcc`:

```sh
cmake -S . -B build-host -DDL_CUDA_ENABLE_CUDA=OFF
cmake --build build-host -j
ctest --test-dir build-host --output-on-failure
```

## CLI

```sh
./build/dl-cuda train-xor --epochs 3000 --lr 0.1
./build/dl-cuda train-char --epochs 800 --print-every 50
./build/dl-cuda sample-char --checkpoint char_v2.ckpt --gen-len 200
```

Options may be passed as `--name value` or `--name=value`.

Standalone wrappers are also built with CUDA enabled:

```sh
./build/dl-cuda-xor --epochs 3000 --lr 0.1
./build/dl-cuda-char-lm --epochs 800 --print-every 50
```

Use config files (key=value) with any subcommand:

```sh
./build/dl-cuda train-char --config configs/char_train.cfg
```

Config keys now match CLI option names without the leading `--`. Example:

```ini
seq-len=64
print-every=50
use-cublas=false
tf32=false
save=false
```

## Programmatic API

Core library:

```cpp
#include "dl_cuda.hpp"

int main() {
  dlcuda::Status status = dlcuda::Status::Ok();
  return status.ok() ? 0 : 1;
}
```

Example workflows:

```cpp
#include "dl_cuda_examples.hpp"

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

## Tests

- `v2_host_tests`: host-only checks for `Status`, `Result`, and `CharVocab`
- `v2_cli_tests`: host-only checks for the shared CLI/config parser
- `v2_cuda_smoke_tests`: CUDA smoke test for the v2 forward/backward path
- legacy v1 tests are opt-in with `-DDL_CUDA_BUILD_LEGACY_TESTS=ON`

For a warnings-as-errors build on machines without `nvcc`, use:

```sh
./scripts/check_no_warnings_build.sh build-warnings host
```

## License

MIT
