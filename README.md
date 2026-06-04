# dl-cuda

A GPU-accelerated deep learning framework in CUDA C++.

## Architecture

This repository exposes a single API:

- `Tensor` + typed shapes/dtypes (`float32`, `float16`, `bfloat16`, `int32`) instead of raw pointer I/O
- Real mixed-precision CUDA paths for tensor ops, modules, losses, and optimizer updates
- Optimizers for SGD, momentum SGD, Adam, AdamW, and RMSProp with parameter groups,
  weight decay, learning-rate schedulers, and optimizer-state checkpoints
- Basic tensor ops: broadcast elementwise add/subtract/multiply/divide, 2D matmul, sum reduction,
  and storage-sharing reshape views
- `AutoTensor`/`GradientTape` autograd for tensor-op graphs, custom registered ops, and wrapping
  manual `Module::Forward`/`Backward` implementations as tape nodes
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
./build/dl-cuda train-char --data data/corpus.txt --epochs 800
./build/dl-cuda train-char --data data/corpus.txt --val-fraction 0.1 --test-fraction 0.1
./build/dl-cuda sample-char --checkpoint char.ckpt --gen-len 200
./build/dl-cuda sample-char --checkpoint char.best.ckpt --prompt "To be" --gen-len 200
./build/dl-cuda sample-char --checkpoint char.ckpt --data data/corpus.txt --gen-len 200
```

Options may be passed as `--name value` or `--name=value`.

`train-char` and `sample-char` default to the embedded demo corpus. Pass `--data PATH` to train or
sample with a text file. Use the same corpus when sampling a custom checkpoint; checkpoints record
corpus/vocabulary fingerprints, and `sample-char` validates them before generating.

`train-char` splits eligible corpus windows into train/validation/test ranges with
`--val-fraction` and `--test-fraction`. It records metric history in checkpoint metadata, evaluates
validation every `--val-every` epochs, saves the validation-selected model to `--best-checkpoint`
(default derived from `--checkpoint`, such as `char.best.ckpt`), can stop early with
`--early-stop-patience`, and reports test metrics on the selected model. `sample-char` and the
programmatic `SampleChar` API also accept `--prompt` / `SampleCharConfig::prompt`.

Use config files (key=value) with any subcommand:

```sh
./build/dl-cuda train-char --config configs/char_train.cfg
```

Config keys now match CLI option names without the leading `--`. Example:

```ini
seq-len=64
data=data/corpus.txt
print-every=50
val-fraction=0.1
test-fraction=0.1
best-checkpoint=char.best.ckpt
prompt=To be
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

Checkpoints store:

- format/version metadata
- model name
- named tensors (name, dtype, shape, raw bytes)
- epoch/step progress
- training config, scheduler state, RNG state, metric history, split metadata, and extra metadata
- corpus and vocabulary metadata for char-language-model runs
- embedded optimizer state when saved through the optimizer-aware overload

`LoadCheckpoint(...)` validates model name + tensor schema before loading. Optimizer-aware loads
also require optimizer state so `--resume` does not silently continue from weights only.

Optimizers also expose `SaveCheckpoint(...)` / `LoadCheckpoint(...)` for stateful resume. Optimizer
checkpoints store the optimizer type, hyperparameters, step count, parameter groups, and
per-parameter optimizer state tensors.

## Tests

- `host_tests`: host-only checks for `Status`, `Result`, and `CharVocab`
- `cli_tests`: host-only checks for the shared CLI/config parser
- `gpu_correctness_tests`: GPU checks for tensor ops, mixed precision, layers, losses, optimizers,
  checkpoints, and autograd
- `char_lm_smoke`: GPU CLI smoke test for `dl-cuda train-char` training/evaluation/generation

For a warnings-as-errors build on machines without `nvcc`, use:

```sh
./scripts/check_no_warnings_build.sh build-warnings host
```

## License

MIT
