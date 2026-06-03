#pragma once

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace cli_parse {

inline constexpr const char kConfigFlag[] = "--config";
inline constexpr const char kConfigInlinePrefix[] = "--config=";
inline constexpr const char kHelpFlag[] = "--help";

inline bool parse_int(const char *text, int *out) {
  if (!text || !out)
    return false;
  errno = 0;
  char *end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0')
    return false;
  if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
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
  if (text[0] == '-')
    return false;
  errno = 0;
  char *end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0')
    return false;
  if (value > static_cast<unsigned long long>(std::numeric_limits<uint64_t>::max()))
    return false;
  *out = static_cast<uint64_t>(value);
  return true;
}

inline bool parse_bool(const std::string &text, bool *out) {
  if (out == nullptr) {
    return false;
  }
  std::string normalized;
  normalized.reserve(text.size());
  for (unsigned char ch : text) {
    normalized.push_back(static_cast<char>(std::tolower(ch)));
  }
  if (normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes") {
    *out = true;
    return true;
  }
  if (normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no") {
    *out = false;
    return true;
  }
  return false;
}

inline std::string trim(const std::string &input) {
  size_t start = input.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = input.find_last_not_of(" \t\r\n");
  return input.substr(start, end - start + 1);
}

inline bool starts_with(const std::string &text, const char *prefix) {
  return text.rfind(prefix, 0) == 0;
}

inline bool split_inline_value(const std::string &arg, std::string *flag, std::string *value) {
  if (flag == nullptr || value == nullptr) {
    return false;
  }
  size_t eq = arg.find('=');
  if (eq == std::string::npos) {
    return false;
  }
  *flag = arg.substr(0, eq);
  *value = arg.substr(eq + 1);
  return true;
}

inline bool load_config_file(const std::string &path,
                             std::unordered_map<std::string, std::string> *out_map,
                             std::string *error) {
  if (out_map == nullptr || error == nullptr) {
    return false;
  }
  out_map->clear();
  std::ifstream in(path);
  if (!in) {
    *error = "failed to open config file: " + path;
    return false;
  }

  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    std::string trimmed = trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }

    size_t eq = trimmed.find('=');
    if (eq == std::string::npos) {
      *error = "invalid config line " + std::to_string(line_no) + ": expected key=value";
      return false;
    }

    std::string key = trim(trimmed.substr(0, eq));
    std::string value = trim(trimmed.substr(eq + 1));
    if (key.empty()) {
      *error = "invalid config line " + std::to_string(line_no) + ": empty key";
      return false;
    }
    (*out_map)[key] = value;
  }

  return true;
}

inline bool get_config_path(int argc, char **argv, int start_index, std::string *path_out,
                            std::string *error) {
  if (path_out == nullptr || error == nullptr) {
    return false;
  }
  path_out->clear();
  if (argc < 0 || start_index < 0 || start_index > argc) {
    *error = "invalid argument range";
    return false;
  }
  if (argv == nullptr && start_index < argc) {
    *error = "argv is null";
    return false;
  }
  for (int i = start_index; i < argc; ++i) {
    if (argv[i] == nullptr) {
      *error = "argv contains null entry";
      return false;
    }
    std::string arg = argv[i];
    if (arg == kConfigFlag) {
      if (i + 1 >= argc) {
        *error = "--config requires a file path";
        return false;
      }
      if (argv[i + 1] == nullptr) {
        *error = "--config file path is null";
        return false;
      }
      *path_out = argv[i + 1];
      return true;
    }
    if (starts_with(arg, kConfigInlinePrefix)) {
      *path_out = arg.substr(sizeof(kConfigInlinePrefix) - 1);
      if (path_out->empty()) {
        *error = "--config requires a file path";
        return false;
      }
      return true;
    }
  }
  return true;
}

inline bool has_help_arg(int argc, char **argv, int start_index, bool *has_help,
                         std::string *error) {
  if (has_help == nullptr || error == nullptr) {
    return false;
  }
  *has_help = false;
  if (argc < 0 || start_index < 0 || start_index > argc) {
    *error = "invalid argument range";
    return false;
  }
  if (argv == nullptr && start_index < argc) {
    *error = "argv is null";
    return false;
  }
  for (int i = start_index; i < argc; ++i) {
    if (argv[i] == nullptr) {
      *error = "argv contains null entry";
      return false;
    }
    if (std::string(argv[i]) == kHelpFlag) {
      *has_help = true;
      return true;
    }
  }
  return true;
}

enum class ParseResult {
  kOk,
  kHelp,
  kError,
};

template <typename Config> struct OptionSpec {
  const char *flag = nullptr;
  const char *config_key = nullptr;
  bool cli_takes_value = false;
  std::function<bool(Config *, const std::string &, std::string *)> set_from_text;
  std::function<void(Config *)> set_from_flag;
};

template <typename Config> inline std::string option_config_key(const OptionSpec<Config> &spec) {
  if (spec.config_key != nullptr) {
    return spec.config_key;
  }
  std::string key = spec.flag ? spec.flag : "";
  if (key.rfind("--", 0) == 0) {
    key.erase(0, 2);
  }
  return key;
}

template <typename Config>
inline const OptionSpec<Config> *find_option_by_flag(const std::vector<OptionSpec<Config>> &options,
                                                     const std::string &flag) {
  for (const auto &option : options) {
    if (option.flag != nullptr && flag == option.flag) {
      return &option;
    }
  }
  return nullptr;
}

template <typename Config>
inline const OptionSpec<Config> *
find_option_by_config_key(const std::vector<OptionSpec<Config>> &options, const std::string &key) {
  for (const auto &option : options) {
    if (key == option_config_key(option)) {
      return &option;
    }
  }
  return nullptr;
}

template <typename Config>
inline bool apply_config_map(const std::unordered_map<std::string, std::string> &cfg_map,
                             const std::vector<OptionSpec<Config>> &options, Config *cfg,
                             std::string *error) {
  if (cfg == nullptr || error == nullptr) {
    return false;
  }
  for (const auto &entry : cfg_map) {
    const OptionSpec<Config> *option = find_option_by_config_key(options, entry.first);
    if (option == nullptr) {
      *error = "unknown config key: " + entry.first;
      return false;
    }
    if (!option->set_from_text) {
      *error = "config key does not accept a value: " + entry.first;
      return false;
    }
    if (!option->set_from_text(cfg, entry.second, error)) {
      return false;
    }
  }
  return true;
}

template <typename Config>
inline ParseResult apply_command_line(int argc, char **argv, int start_index,
                                      const std::vector<OptionSpec<Config>> &options, Config *cfg,
                                      std::string *error) {
  if (cfg == nullptr || error == nullptr) {
    return ParseResult::kError;
  }
  if (argc < 0 || start_index < 0 || start_index > argc) {
    *error = "invalid argument range";
    return ParseResult::kError;
  }
  if (argv == nullptr && start_index < argc) {
    *error = "argv is null";
    return ParseResult::kError;
  }
  for (int i = start_index; i < argc; ++i) {
    if (argv[i] == nullptr) {
      *error = "argv contains null entry";
      return ParseResult::kError;
    }
    std::string arg = argv[i];
    if (arg == kHelpFlag) {
      return ParseResult::kHelp;
    }
    if (arg == kConfigFlag) {
      if (i + 1 >= argc) {
        *error = "--config requires a file path";
        return ParseResult::kError;
      }
      if (argv[i + 1] == nullptr) {
        *error = "--config file path is null";
        return ParseResult::kError;
      }
      ++i;
      continue;
    }
    if (starts_with(arg, kConfigInlinePrefix)) {
      if (arg.size() == sizeof(kConfigInlinePrefix) - 1) {
        *error = "--config requires a file path";
        return ParseResult::kError;
      }
      continue;
    }

    std::string flag = arg;
    std::string inline_value;
    bool has_inline_value = split_inline_value(arg, &flag, &inline_value);

    const OptionSpec<Config> *option = find_option_by_flag(options, flag);
    if (option == nullptr) {
      *error = "unknown option: " + arg;
      return ParseResult::kError;
    }

    if (option->cli_takes_value) {
      if (!option->set_from_text) {
        *error = "option does not accept a value: " + flag;
        return ParseResult::kError;
      }
      if (has_inline_value) {
        if (!option->set_from_text(cfg, inline_value, error)) {
          return ParseResult::kError;
        }
        continue;
      }
      if (i + 1 >= argc) {
        *error = "missing value for " + flag;
        return ParseResult::kError;
      }
      if (argv[i + 1] == nullptr) {
        *error = "null value for " + flag;
        return ParseResult::kError;
      }
      if (!option->set_from_text(cfg, argv[++i], error)) {
        return ParseResult::kError;
      }
      continue;
    }

    if (has_inline_value) {
      *error = "option does not accept an inline value: " + flag;
      return ParseResult::kError;
    }
    if (!option->set_from_flag) {
      *error = "option requires a value: " + flag;
      return ParseResult::kError;
    }
    option->set_from_flag(cfg);
  }

  return ParseResult::kOk;
}

template <typename Config, typename PrintUsageFn>
inline int configure_command(int argc, char **argv, int start_index,
                             const std::vector<OptionSpec<Config>> &options, Config *cfg,
                             const PrintUsageFn &print_usage) {
  if (cfg == nullptr) {
    std::fprintf(stderr, "Config target is null\n");
    return 1;
  }

  std::string config_path;
  std::string error;
  bool has_help = false;
  if (!has_help_arg(argc, argv, start_index, &has_help, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }
  if (has_help) {
    print_usage();
    return 0;
  }

  if (!get_config_path(argc, argv, start_index, &config_path, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }

  std::unordered_map<std::string, std::string> cfg_map;
  if (!config_path.empty() && !load_config_file(config_path, &cfg_map, &error)) {
    std::fprintf(stderr, "Config error: %s\n", error.c_str());
    return 1;
  }

  if (!apply_config_map(cfg_map, options, cfg, &error)) {
    std::fprintf(stderr, "Config error: %s\n", error.c_str());
    return 1;
  }

  ParseResult parse_result = apply_command_line(argc, argv, start_index, options, cfg, &error);
  if (parse_result == ParseResult::kHelp) {
    print_usage();
    return 0;
  }
  if (parse_result == ParseResult::kError) {
    std::fprintf(stderr, "%s\n", error.c_str());
    print_usage();
    return 1;
  }

  return -1;
}

template <typename Config>
inline OptionSpec<Config> int_option(const char *flag, int Config::*field,
                                     const char *config_key = nullptr) {
  return OptionSpec<Config>{
      flag,
      config_key,
      true,
      [field, flag](Config *cfg, const std::string &text, std::string *error) {
        int value = 0;
        if (!parse_int(text.c_str(), &value)) {
          *error = "invalid integer for " + std::string(flag);
          return false;
        }
        cfg->*field = value;
        return true;
      },
      {},
  };
}

template <typename Config>
inline OptionSpec<Config> float_option(const char *flag, float Config::*field,
                                       const char *config_key = nullptr) {
  return OptionSpec<Config>{
      flag,
      config_key,
      true,
      [field, flag](Config *cfg, const std::string &text, std::string *error) {
        float value = 0.0f;
        if (!parse_float(text.c_str(), &value)) {
          *error = "invalid float for " + std::string(flag);
          return false;
        }
        cfg->*field = value;
        return true;
      },
      {},
  };
}

template <typename Config>
inline OptionSpec<Config> u64_option(const char *flag, uint64_t Config::*field,
                                     const char *config_key = nullptr) {
  return OptionSpec<Config>{
      flag,
      config_key,
      true,
      [field, flag](Config *cfg, const std::string &text, std::string *error) {
        uint64_t value = 0;
        if (!parse_u64(text.c_str(), &value)) {
          *error = "invalid uint64 for " + std::string(flag);
          return false;
        }
        cfg->*field = value;
        return true;
      },
      {},
  };
}

template <typename Config>
inline OptionSpec<Config> string_option(const char *flag, std::string Config::*field,
                                        const char *config_key = nullptr) {
  return OptionSpec<Config>{
      flag,
      config_key,
      true,
      [field](Config *cfg, const std::string &text, std::string *error) {
        (void)error;
        cfg->*field = text;
        return true;
      },
      {},
  };
}

template <typename Config>
inline OptionSpec<Config> bool_flag_option(const char *flag, bool Config::*field, bool flag_value,
                                           const char *config_key = nullptr) {
  return OptionSpec<Config>{
      flag,
      config_key,
      false,
      [field, flag](Config *cfg, const std::string &text, std::string *error) {
        bool value = false;
        if (!parse_bool(text, &value)) {
          *error = "invalid bool for " + std::string(flag);
          return false;
        }
        cfg->*field = value;
        return true;
      },
      [field, flag_value](Config *cfg) { cfg->*field = flag_value; },
  };
}

} // namespace cli_parse
