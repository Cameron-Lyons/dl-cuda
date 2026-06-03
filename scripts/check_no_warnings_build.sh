#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"
CUDA_MODE="${2:-auto}"

case "$CUDA_MODE" in
  auto)
    if command -v nvcc >/dev/null 2>&1; then
      CUDA_FLAG=ON
    else
      CUDA_FLAG=OFF
    fi
    ;;
  cuda)
    CUDA_FLAG=ON
    ;;
  host)
    CUDA_FLAG=OFF
    ;;
  *)
    echo "Usage: $0 [build-dir] [auto|cuda|host]" >&2
    exit 1
    ;;
esac

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DDL_CUDA_WARNINGS_AS_ERRORS=ON \
  -DDL_CUDA_ENABLE_CUDA="$CUDA_FLAG"
if ! cmake --build "$BUILD_DIR" -j >"$BUILD_DIR/build.log" 2>&1; then
  echo "Build failed. Log follows:" >&2
  cat "$BUILD_DIR/build.log" >&2
  exit 1
fi

if rg -n "warning:" "$BUILD_DIR/build.log" >/dev/null 2>&1; then
  echo "Build produced warnings. Failing no-warnings check." >&2
  rg -n "warning:" "$BUILD_DIR/build.log" >&2
  exit 1
fi

echo "No-warnings build check: PASS"
