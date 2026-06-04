#!/usr/bin/env bash
set -euo pipefail

BIN="${DL_CUDA_BIN:-./build/dl-cuda}"
EPOCHS="${1:-${EPOCHS:-200}}"
SEQ_LEN="${2:-${SEQ_LEN:-64}}"
PRINT_EVERY="${PRINT_EVERY:-$EPOCHS}"

if [[ "$PRINT_EVERY" =~ ^[0-9]+$ ]] && [ "$PRINT_EVERY" -lt 1 ]; then
  PRINT_EVERY=1
fi

if [ ! -x "$BIN" ]; then
  echo "Missing executable: $BIN" >&2
  echo "Build with CUDA enabled first: cmake -S . -B build && cmake --build build -j" >&2
  exit 1
fi

"$BIN" \
  train-char \
  --epochs "$EPOCHS" \
  --seq-len "$SEQ_LEN" \
  --print-every "$PRINT_EVERY" \
  --gen-len 0 \
  --no-save
