#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
case "$MODE" in
  check | fix)
    ;;
  *)
    echo "Usage: $0 check|fix" >&2
    exit 2
    ;;
esac

if command -v clang-format >/dev/null 2>&1; then
  CLANG_FORMAT=clang-format
elif command -v clang-format-18 >/dev/null 2>&1; then
  CLANG_FORMAT=clang-format-18
elif command -v clang-format-17 >/dev/null 2>&1; then
  CLANG_FORMAT=clang-format-17
else
  echo "clang-format is required but was not found on PATH" >&2
  exit 1
fi

FILES=()
while IFS= read -r -d '' file; do
  FILES+=("$file")
done < <(rg --files \
  examples include tests src \
  -g '*.cu' -g '*.cuh' -g '*.h' -g '*.hpp' -g '*.cpp' -g '*.cc' \
  -g '!build/**' -g '!profiles/**' -0)

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No source files to format"
  exit 0
fi

if [ "$MODE" = "check" ]; then
  "$CLANG_FORMAT" --dry-run --Werror "${FILES[@]}"
  echo "clang-format lint: PASS"
else
  "$CLANG_FORMAT" -i "${FILES[@]}"
  echo "clang-format apply: DONE"
fi
