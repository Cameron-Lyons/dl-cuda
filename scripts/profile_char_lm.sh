#!/usr/bin/env bash
set -euo pipefail

mkdir -p profiles

PROFILE_MODE="${PROFILE_MODE:-train}"   # train | e2e
CUDA_GRAPH_MODE="${CUDA_GRAPH_MODE:-on}" # on | off
EPOCHS="${EPOCHS:-80}"
PRINT_EVERY="${PRINT_EVERY:-20}"
GEN_LEN="${GEN_LEN:-32}"

RUN_ARGS=(--epochs "${EPOCHS}")

case "${PROFILE_MODE}" in
  train)
    RUN_ARGS+=(--print-every "${PRINT_EVERY}" --gen-len 0 --no-save --no-train-metrics)
    ;;
  e2e)
    RUN_ARGS+=(--print-every "${PRINT_EVERY}" --gen-len "${GEN_LEN}")
    ;;
  *)
    echo "Unsupported PROFILE_MODE='${PROFILE_MODE}'. Use 'train' or 'e2e'." >&2
    exit 2
    ;;
esac

if [[ "${CUDA_GRAPH_MODE}" == "off" ]]; then
  RUN_ARGS+=(--no-cuda-graph)
elif [[ "${CUDA_GRAPH_MODE}" != "on" ]]; then
  echo "Unsupported CUDA_GRAPH_MODE='${CUDA_GRAPH_MODE}'. Use 'on' or 'off'." >&2
  exit 2
fi

NSYS_OUT="profiles/char_lm_${PROFILE_MODE}_graph-${CUDA_GRAPH_MODE}_$(date +%Y%m%d_%H%M%S)"

nsys profile \
  --sample=none \
  --trace=cuda,nvtx,osrt \
  --stats=true \
  -o "${NSYS_OUT}" \
  ./build/dl-cuda-char-lm "${RUN_ARGS[@]}"

echo "Wrote profile report: ${NSYS_OUT}.nsys-rep"
