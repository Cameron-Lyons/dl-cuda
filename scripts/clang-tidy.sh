#!/usr/bin/env bash
# Run clang-tidy on project sources.
# With compile_commands.json (after cmake in build/): runs on all sources.
# Without it (e.g. no CUDA/nvcc): runs only on C++-only headers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

if command -v clang-tidy >/dev/null 2>&1; then
  CLANG_TIDY=clang-tidy
elif command -v clang-tidy-18 >/dev/null 2>&1; then
  CLANG_TIDY=clang-tidy-18
elif command -v clang-tidy-17 >/dev/null 2>&1; then
  CLANG_TIDY=clang-tidy-17
else
  echo "clang-tidy is required but was not found on PATH" >&2
  exit 1
fi

ALL_FILES=()
while IFS= read -r line; do
  [ -n "$line" ] && ALL_FILES+=("$line")
done < <(rg --files examples include tests src \
  -g '*.cu' -g '*.cuh' -g '*.h' -g '*.hpp' -g '*.cpp' -g '*.cc' \
  -g '!build/**' -g '!profiles/**' 2>/dev/null || true)

if [ ${#ALL_FILES[@]} -eq 0 ]; then
  echo "No source files to lint"
  exit 0
fi

if [ -f build/compile_commands.json ]; then
  echo "Using build/compile_commands.json"
  for f in "${ALL_FILES[@]}"; do
    "$CLANG_TIDY" "$f" -p build || true
  done
  echo "clang-tidy: done"
  exit 0
fi

SYSROOT=""
if [ "$(uname -s)" = "Darwin" ]; then
  SYSROOT="$(xcrun --show-sdk-path 2>/dev/null || true)"
fi
EXTRA=(-I "$ROOT/include" -I "$ROOT/src" -std=c++17)
[ -n "$SYSROOT" ] && EXTRA=(-isysroot "$SYSROOT" "${EXTRA[@]}")

# C++-only headers that do not include CUDA (so we can run without nvcc)
CPP_ONLY=(include/dl_cuda/examples.hpp include/dl_cuda.hpp examples/cli_parse_utils.hpp)
for f in "${CPP_ONLY[@]}"; do
  [ -f "$f" ] && "$CLANG_TIDY" "$f" -- "${EXTRA[@]}" || true
done
echo "Note: Full clang-tidy (CUDA sources) requires a configured build: mkdir build && cd build && cmake .." >&2
echo "clang-tidy: done"
