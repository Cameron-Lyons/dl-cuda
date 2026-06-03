#!/usr/bin/env bash
set -euo pipefail

mkdir -p profiles

BIN="${DL_CUDA_CHAR_BIN:-./build/dl-cuda-char-lm}"
PROFILE_MODE="${PROFILE_MODE:-train}"   # train | e2e
EPOCHS="${EPOCHS:-80}"
PRINT_EVERY="${PRINT_EVERY:-20}"
GEN_LEN="${GEN_LEN:-32}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys is required but was not found on PATH" >&2
  exit 1
fi

if [ ! -x "$BIN" ]; then
  echo "Missing executable: $BIN" >&2
  echo "Build with CUDA enabled first: cmake -S . -B build && cmake --build build -j" >&2
  exit 1
fi

RUN_ARGS=(--epochs "${EPOCHS}")

case "${PROFILE_MODE}" in
  train)
    RUN_ARGS+=(--print-every "${PRINT_EVERY}" --gen-len 0 --no-save)
    ;;
  e2e)
    RUN_ARGS+=(--print-every "${PRINT_EVERY}" --gen-len "${GEN_LEN}")
    ;;
  *)
    echo "Unsupported PROFILE_MODE='${PROFILE_MODE}'. Use 'train' or 'e2e'." >&2
    exit 2
    ;;
esac

NSYS_OUT="profiles/char_lm_${PROFILE_MODE}_$(date +%Y%m%d_%H%M%S)"

nsys profile \
  --sample=none \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  -o "${NSYS_OUT}" \
  "$BIN" "${RUN_ARGS[@]}"

echo "Wrote profile report: ${NSYS_OUT}.nsys-rep"
