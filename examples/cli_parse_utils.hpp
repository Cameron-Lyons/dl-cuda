#pragma once

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace cli_parse {

inline bool parse_int(const char *text, int *out) {
  if (!text || !out)
    return false;
  errno = 0;
  char *end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0')
    return false;
  if (value < std::numeric_limits<int>::min() ||
      value > std::numeric_limits<int>::max())
    return false;
  *out = static_cast<int>(value);
  return true;
}

inline bool parse_float(const char *text, float *out) {
  if (!text || !out)
    return false;
  errno = 0;
  char *end = nullptr;
  float value = std::strtof(text, &end);
  if (errno != 0 || end == text || *end != '\0')
    return false;
  if (!std::isfinite(value))
    return false;
  *out = value;
  return true;
}

inline bool parse_u64(const char *text, uint64_t *out) {
  if (!text || !out)
    return false;
  errno = 0;
  char *end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0')
    return false;
  *out = static_cast<uint64_t>(value);
  return true;
}

template <typename PrintUsageFn>
inline int invalid_value(const char *option, const char *value,
                         const PrintUsageFn &print_usage) {
  std::fprintf(stderr, "Invalid value for %s: %s\n", option,
               value ? value : "<missing>");
  print_usage();
  return 1;
}

} // namespace cli_parse
