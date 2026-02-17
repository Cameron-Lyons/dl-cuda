#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-200}"
SEQ_LEN="${2:-64}"

./build/dl-cuda-char-lm \
  --epochs "$EPOCHS" \
  --seq-len "$SEQ_LEN" \
  --print-every "$EPOCHS" \
  --gen-len 0 \
  --no-save
