# dl-cuda

A GPU-accelerated deep learning framework written from scratch in CUDA C++.

## Features

**Layers**
- Linear (fully connected)
- Conv1D / Conv2D
- LSTM
- Elman RNN
- Transformer (multi-head attention, layer norm, residual connections)
- Dropout

**Activations** — ReLU, Sigmoid, Tanh

**Loss Functions** — MSE, MAE, Binary Cross-Entropy, Categorical Cross-Entropy

**Optimizers** — SGD, RMSprop, Adam

**Metrics** — R², Accuracy, F1 Score, Matthews Correlation Coefficient

**Utilities** — Tensor operations, CSV data loading, Sequential model API

## Requirements

- CMake 3.18+
- NVIDIA CUDA Toolkit
- C++17 compiler

## Build

```sh
cmake -B build
cmake --build build
```

## Usage

The included example trains a two-layer network on the XOR problem:

```sh
./build/dl-cuda
```

```
Epoch 0, MSE Loss: 0.372253
Epoch 500, MSE Loss: 0.001204
...

Final predictions:
  [0, 0] -> 0.0197 (expected 0)
  [0, 1] -> 0.9718 (expected 1)
  [1, 0] -> 0.9726 (expected 1)
  [1, 1] -> 0.0362 (expected 0)
```

## License

MIT
