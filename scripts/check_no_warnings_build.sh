#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${1:-build}"

cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DDL_CUDA_WARNINGS_AS_ERRORS=ON
cmake --build "$BUILD_DIR" -j >"$BUILD_DIR/build.log" 2>&1

if rg -n "warning:" "$BUILD_DIR/build.log" >/dev/null 2>&1; then
  echo "Build produced warnings. Failing no-warnings check." >&2
  rg -n "warning:" "$BUILD_DIR/build.log" >&2
  exit 1
fi

echo "No-warnings build check: PASS"
