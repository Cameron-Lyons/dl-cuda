#!/usr/bin/env bash
set -euo pipefail

mkdir -p profiles

NSYS_OUT="profiles/char_lm_$(date +%Y%m%d_%H%M%S)"

nsys profile \
  --sample=none \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  -o "${NSYS_OUT}" \
  ./build/dl-cuda-char-lm --epochs 80 --print-every 20 --gen-len 32

echo "Wrote profile report: ${NSYS_OUT}.nsys-rep"
