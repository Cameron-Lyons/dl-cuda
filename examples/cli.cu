#include "dl_cuda/examples.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace {

bool ParseInt(const char *text, int *out) {
  if (!text || !out) {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    return false;
  }
  *out = static_cast<int>(value);
  return true;
}

bool ParseFloat(const char *text, float *out) {
  if (!text || !out) {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  float value = std::strtof(text, &end);
  if (errno != 0 || end == text || *end != '\0') {
    return false;
  }
  *out = value;
  return true;
}

bool ParseU64(const char *text, uint64_t *out) {
  if (!text || !out) {
    return false;
  }
  errno = 0;
  char *end = nullptr;
  unsigned long long value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0') {
    return false;
  }
  *out = static_cast<uint64_t>(value);
  return true;
}

bool ParseBool(const std::string &text, bool *out) {
  if (text == "1" || text == "true" || text == "on" || text == "yes") {
    *out = true;
    return true;
  }
  if (text == "0" || text == "false" || text == "off" || text == "no") {
    *out = false;
    return true;
  }
  return false;
}

std::string Trim(const std::string &input) {
  size_t start = input.find_first_not_of(" \t\r\n");
  if (start == std::string::npos) {
    return "";
  }
  size_t end = input.find_last_not_of(" \t\r\n");
  return input.substr(start, end - start + 1);
}

bool LoadConfigFile(const std::string &path,
                    std::unordered_map<std::string, std::string> *out_map,
                    std::string *error) {
  std::ifstream in(path);
  if (!in) {
    *error = "failed to open config file: " + path;
    return false;
  }

  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed[0] == '#') {
      continue;
    }

    size_t eq = trimmed.find('=');
    if (eq == std::string::npos) {
      *error = "invalid config line " + std::to_string(line_no) +
               ": expected key=value";
      return false;
    }

    std::string key = Trim(trimmed.substr(0, eq));
    std::string value = Trim(trimmed.substr(eq + 1));
    if (key.empty()) {
      *error = "invalid config line " + std::to_string(line_no) + ": empty key";
      return false;
    }

    (*out_map)[key] = value;
  }

  return true;
}

void PrintUsage() {
  std::puts("Usage: dl-cuda <subcommand> [options]");
  std::puts("Subcommands:");
  std::puts("  train-xor   Train XOR MLP model");
  std::puts("  train-char  Train char-level model");
  std::puts("  sample-char Generate text from char-level checkpoint");
  std::puts("\nCommon options:");
  std::puts("  --config PATH     Load key=value config file");
  std::puts("\nRun 'dl-cuda <subcommand> --help' for subcommand options.");
}

void PrintTrainXorUsage() {
  std::puts("Usage: dl-cuda train-xor [options]");
  std::puts("  --config PATH");
  std::puts("  --epochs N");
  std::puts("  --print-every N");
  std::puts("  --hidden-size N");
  std::puts("  --lr F");
  std::puts("  --grad-clip F");
  std::puts("  --seed N");
  std::puts("  --checkpoint PATH");
  std::puts("  --resume");
  std::puts("  --no-save");
  std::puts("  --no-cublas");
  std::puts("  --no-tf32");
}

void PrintTrainCharUsage() {
  std::puts("Usage: dl-cuda train-char [options]");
  std::puts("  --config PATH");
  std::puts("  --seq-len N");
  std::puts("  --d-model N");
  std::puts("  --epochs N");
  std::puts("  --print-every N");
  std::puts("  --lr F");
  std::puts("  --grad-clip F");
  std::puts("  --temperature F");
  std::puts("  --top-p F");
  std::puts("  --gen-len N");
  std::puts("  --seed N");
  std::puts("  --sample-seed N");
  std::puts("  --checkpoint PATH");
  std::puts("  --resume");
  std::puts("  --no-save");
  std::puts("  --no-cublas");
  std::puts("  --no-tf32");
}

void PrintSampleCharUsage() {
  std::puts("Usage: dl-cuda sample-char [options]");
  std::puts("  --config PATH");
  std::puts("  --seq-len N");
  std::puts("  --d-model N");
  std::puts("  --gen-len N");
  std::puts("  --temperature F");
  std::puts("  --top-p F");
  std::puts("  --seed N");
  std::puts("  --sample-seed N");
  std::puts("  --checkpoint PATH");
  std::puts("  --no-cublas");
  std::puts("  --no-tf32");
}

bool GetConfigPath(int argc, char **argv, int start_index, std::string *path_out,
                   std::string *error) {
  for (int i = start_index; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--config") {
      if (i + 1 >= argc) {
        *error = "--config requires a file path";
        return false;
      }
      *path_out = argv[i + 1];
      return true;
    }
  }
  return true;
}

bool ApplyXorConfigMap(const std::unordered_map<std::string, std::string> &cfg_map,
                       dlcuda::TrainXorConfig *cfg, std::string *error) {
  auto apply_int = [&](const char *key, int *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseInt(it->second.c_str(), field)) {
      *error = std::string("invalid integer for ") + key;
      return false;
    }
    return true;
  };
  auto apply_float = [&](const char *key, float *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseFloat(it->second.c_str(), field)) {
      *error = std::string("invalid float for ") + key;
      return false;
    }
    return true;
  };
  auto apply_u64 = [&](const char *key, uint64_t *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseU64(it->second.c_str(), field)) {
      *error = std::string("invalid uint64 for ") + key;
      return false;
    }
    return true;
  };
  auto apply_bool = [&](const char *key, bool *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseBool(it->second, field)) {
      *error = std::string("invalid bool for ") + key;
      return false;
    }
    return true;
  };

  if (!apply_int("epochs", &cfg->epochs))
    return false;
  if (!apply_int("print_every", &cfg->print_every))
    return false;
  if (!apply_int("hidden_size", &cfg->hidden_size))
    return false;
  if (!apply_float("lr", &cfg->lr))
    return false;
  if (!apply_float("grad_clip", &cfg->grad_clip))
    return false;
  if (!apply_u64("seed", &cfg->seed))
    return false;
  if (!apply_bool("resume", &cfg->resume))
    return false;
  if (!apply_bool("save", &cfg->save))
    return false;
  if (!apply_bool("use_cublas", &cfg->use_cublas))
    return false;
  if (!apply_bool("tf32", &cfg->tf32))
    return false;

  auto ckpt = cfg_map.find("checkpoint");
  if (ckpt != cfg_map.end()) {
    cfg->checkpoint_path = ckpt->second;
  }

  return true;
}

bool ApplyTrainCharConfigMap(const std::unordered_map<std::string, std::string> &cfg_map,
                             dlcuda::TrainCharConfig *cfg, std::string *error) {
  auto apply_int = [&](const char *key, int *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseInt(it->second.c_str(), field)) {
      *error = std::string("invalid integer for ") + key;
      return false;
    }
    return true;
  };
  auto apply_float = [&](const char *key, float *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseFloat(it->second.c_str(), field)) {
      *error = std::string("invalid float for ") + key;
      return false;
    }
    return true;
  };
  auto apply_u64 = [&](const char *key, uint64_t *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseU64(it->second.c_str(), field)) {
      *error = std::string("invalid uint64 for ") + key;
      return false;
    }
    return true;
  };
  auto apply_bool = [&](const char *key, bool *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseBool(it->second, field)) {
      *error = std::string("invalid bool for ") + key;
      return false;
    }
    return true;
  };

  if (!apply_int("seq_len", &cfg->seq_len))
    return false;
  if (!apply_int("d_model", &cfg->d_model))
    return false;
  if (!apply_int("epochs", &cfg->epochs))
    return false;
  if (!apply_int("print_every", &cfg->print_every))
    return false;
  if (!apply_int("gen_len", &cfg->gen_len))
    return false;
  if (!apply_float("lr", &cfg->lr))
    return false;
  if (!apply_float("grad_clip", &cfg->grad_clip))
    return false;
  if (!apply_float("temperature", &cfg->temperature))
    return false;
  if (!apply_float("top_p", &cfg->top_p))
    return false;
  if (!apply_u64("seed", &cfg->seed))
    return false;
  if (!apply_u64("sample_seed", &cfg->sample_seed))
    return false;
  if (!apply_bool("resume", &cfg->resume))
    return false;
  if (!apply_bool("save", &cfg->save))
    return false;
  if (!apply_bool("use_cublas", &cfg->use_cublas))
    return false;
  if (!apply_bool("tf32", &cfg->tf32))
    return false;

  auto ckpt = cfg_map.find("checkpoint");
  if (ckpt != cfg_map.end()) {
    cfg->checkpoint_path = ckpt->second;
  }

  return true;
}

bool ApplySampleCharConfigMap(const std::unordered_map<std::string, std::string> &cfg_map,
                              dlcuda::SampleCharConfig *cfg,
                              std::string *error) {
  auto apply_int = [&](const char *key, int *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseInt(it->second.c_str(), field)) {
      *error = std::string("invalid integer for ") + key;
      return false;
    }
    return true;
  };
  auto apply_float = [&](const char *key, float *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseFloat(it->second.c_str(), field)) {
      *error = std::string("invalid float for ") + key;
      return false;
    }
    return true;
  };
  auto apply_u64 = [&](const char *key, uint64_t *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseU64(it->second.c_str(), field)) {
      *error = std::string("invalid uint64 for ") + key;
      return false;
    }
    return true;
  };
  auto apply_bool = [&](const char *key, bool *field) {
    auto it = cfg_map.find(key);
    if (it == cfg_map.end()) {
      return true;
    }
    if (!ParseBool(it->second, field)) {
      *error = std::string("invalid bool for ") + key;
      return false;
    }
    return true;
  };

  if (!apply_int("seq_len", &cfg->seq_len))
    return false;
  if (!apply_int("d_model", &cfg->d_model))
    return false;
  if (!apply_int("gen_len", &cfg->gen_len))
    return false;
  if (!apply_float("temperature", &cfg->temperature))
    return false;
  if (!apply_float("top_p", &cfg->top_p))
    return false;
  if (!apply_u64("seed", &cfg->seed))
    return false;
  if (!apply_u64("sample_seed", &cfg->sample_seed))
    return false;
  if (!apply_bool("use_cublas", &cfg->use_cublas))
    return false;
  if (!apply_bool("tf32", &cfg->tf32))
    return false;

  auto ckpt = cfg_map.find("checkpoint");
  if (ckpt != cfg_map.end()) {
    cfg->checkpoint_path = ckpt->second;
  }

  return true;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    PrintUsage();
    return 1;
  }

  std::string command = argv[1];
  if (command == "--help" || command == "help") {
    PrintUsage();
    return 0;
  }

  std::string config_path;
  std::string error;
  if (!GetConfigPath(argc, argv, 2, &config_path, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    return 1;
  }

  std::unordered_map<std::string, std::string> cfg_map;
  if (!config_path.empty()) {
    if (!LoadConfigFile(config_path, &cfg_map, &error)) {
      std::fprintf(stderr, "Config error: %s\n", error.c_str());
      return 1;
    }
  }

  if (command == "train-xor") {
    dlcuda::TrainXorConfig cfg;
    if (!ApplyXorConfigMap(cfg_map, &cfg, &error)) {
      std::fprintf(stderr, "Config error: %s\n", error.c_str());
      return 1;
    }

    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--help") {
        PrintTrainXorUsage();
        return 0;
      }
      if (arg == "--config") {
        ++i;
        continue;
      }
      if (arg == "--epochs") {
        if (i + 1 >= argc || !ParseInt(argv[++i], &cfg.epochs)) {
          std::fprintf(stderr, "Invalid value for --epochs\n");
          return 1;
        }
      } else if (arg == "--print-every") {
        if (i + 1 >= argc || !ParseInt(argv[++i], &cfg.print_every)) {
          std::fprintf(stderr, "Invalid value for --print-every\n");
          return 1;
        }
      } else if (arg == "--hidden-size") {
        if (i + 1 >= argc || !ParseInt(argv[++i], &cfg.hidden_size)) {
          std::fprintf(stderr, "Invalid value for --hidden-size\n");
          return 1;
        }
      } else if (arg == "--lr") {
        if (i + 1 >= argc || !ParseFloat(argv[++i], &cfg.lr)) {
          std::fprintf(stderr, "Invalid value for --lr\n");
          return 1;
        }
      } else if (arg == "--grad-clip") {
        if (i + 1 >= argc || !ParseFloat(argv[++i], &cfg.grad_clip)) {
          std::fprintf(stderr, "Invalid value for --grad-clip\n");
          return 1;
        }
      } else if (arg == "--seed") {
        if (i + 1 >= argc || !ParseU64(argv[++i], &cfg.seed)) {
          std::fprintf(stderr, "Invalid value for --seed\n");
          return 1;
        }
      } else if (arg == "--checkpoint") {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "Missing value for --checkpoint\n");
          return 1;
        }
        cfg.checkpoint_path = argv[++i];
      } else if (arg == "--resume") {
        cfg.resume = true;
      } else if (arg == "--no-save") {
        cfg.save = false;
      } else if (arg == "--no-cublas") {
        cfg.use_cublas = false;
      } else if (arg == "--no-tf32") {
        cfg.tf32 = false;
      } else {
        std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
        PrintTrainXorUsage();
        return 1;
      }
    }

    dlcuda::Status status = dlcuda::TrainXor(cfg);
    if (!status.ok()) {
      std::fprintf(stderr, "train-xor failed: %s\n", status.message().c_str());
      return 1;
    }
    return 0;
  }

  if (command == "train-char") {
    dlcuda::TrainCharConfig cfg;
    if (!ApplyTrainCharConfigMap(cfg_map, &cfg, &error)) {
      std::fprintf(stderr, "Config error: %s\n", error.c_str());
      return 1;
    }

    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--help") {
        PrintTrainCharUsage();
        return 0;
      }
      if (arg == "--config") {
        ++i;
        continue;
      }
      if (arg == "--seq-len") {
        if (i + 1 >= argc || !ParseInt(argv[++i], &cfg.seq_len)) {
          std::fprintf(stderr, "Invalid value for --seq-len\n");
          return 1;
        }
      } else if (arg == "--d-model") {
        if (i + 1 >= argc || !ParseInt(argv[++i], &cfg.d_model)) {
          std::fprintf(stderr, "Invalid value for --d-model\n");
          return 1;
        }
      } else if (arg == "--epochs") {
        if (i + 1 >= argc || !ParseInt(argv[++i], &cfg.epochs)) {
          std::fprintf(stderr, "Invalid value for --epochs\n");
          return 1;
        }
      } else if (arg == "--print-every") {
        if (i + 1 >= argc || !ParseInt(argv[++i], &cfg.print_every)) {
          std::fprintf(stderr, "Invalid value for --print-every\n");
          return 1;
        }
      } else if (arg == "--lr") {
        if (i + 1 >= argc || !ParseFloat(argv[++i], &cfg.lr)) {
          std::fprintf(stderr, "Invalid value for --lr\n");
          return 1;
        }
      } else if (arg == "--grad-clip") {
        if (i + 1 >= argc || !ParseFloat(argv[++i], &cfg.grad_clip)) {
          std::fprintf(stderr, "Invalid value for --grad-clip\n");
          return 1;
        }
      } else if (arg == "--temperature") {
        if (i + 1 >= argc || !ParseFloat(argv[++i], &cfg.temperature)) {
          std::fprintf(stderr, "Invalid value for --temperature\n");
          return 1;
        }
      } else if (arg == "--top-p") {
        if (i + 1 >= argc || !ParseFloat(argv[++i], &cfg.top_p)) {
          std::fprintf(stderr, "Invalid value for --top-p\n");
          return 1;
        }
      } else if (arg == "--gen-len") {
        if (i + 1 >= argc || !ParseInt(argv[++i], &cfg.gen_len)) {
          std::fprintf(stderr, "Invalid value for --gen-len\n");
          return 1;
        }
      } else if (arg == "--seed") {
        if (i + 1 >= argc || !ParseU64(argv[++i], &cfg.seed)) {
          std::fprintf(stderr, "Invalid value for --seed\n");
          return 1;
        }
      } else if (arg == "--sample-seed") {
        if (i + 1 >= argc || !ParseU64(argv[++i], &cfg.sample_seed)) {
          std::fprintf(stderr, "Invalid value for --sample-seed\n");
          return 1;
        }
      } else if (arg == "--checkpoint") {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "Missing value for --checkpoint\n");
          return 1;
        }
        cfg.checkpoint_path = argv[++i];
      } else if (arg == "--resume") {
        cfg.resume = true;
      } else if (arg == "--no-save") {
        cfg.save = false;
      } else if (arg == "--no-cublas") {
        cfg.use_cublas = false;
      } else if (arg == "--no-tf32") {
        cfg.tf32 = false;
      } else {
        std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
        PrintTrainCharUsage();
        return 1;
      }
    }

    dlcuda::Status status = dlcuda::TrainChar(cfg);
    if (!status.ok()) {
      std::fprintf(stderr, "train-char failed: %s\n", status.message().c_str());
      return 1;
    }
    return 0;
  }

  if (command == "sample-char") {
    dlcuda::SampleCharConfig cfg;
    if (!ApplySampleCharConfigMap(cfg_map, &cfg, &error)) {
      std::fprintf(stderr, "Config error: %s\n", error.c_str());
      return 1;
    }

    for (int i = 2; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--help") {
        PrintSampleCharUsage();
        return 0;
      }
      if (arg == "--config") {
        ++i;
        continue;
      }
      if (arg == "--seq-len") {
        if (i + 1 >= argc || !ParseInt(argv[++i], &cfg.seq_len)) {
          std::fprintf(stderr, "Invalid value for --seq-len\n");
          return 1;
        }
      } else if (arg == "--d-model") {
        if (i + 1 >= argc || !ParseInt(argv[++i], &cfg.d_model)) {
          std::fprintf(stderr, "Invalid value for --d-model\n");
          return 1;
        }
      } else if (arg == "--gen-len") {
        if (i + 1 >= argc || !ParseInt(argv[++i], &cfg.gen_len)) {
          std::fprintf(stderr, "Invalid value for --gen-len\n");
          return 1;
        }
      } else if (arg == "--temperature") {
        if (i + 1 >= argc || !ParseFloat(argv[++i], &cfg.temperature)) {
          std::fprintf(stderr, "Invalid value for --temperature\n");
          return 1;
        }
      } else if (arg == "--top-p") {
        if (i + 1 >= argc || !ParseFloat(argv[++i], &cfg.top_p)) {
          std::fprintf(stderr, "Invalid value for --top-p\n");
          return 1;
        }
      } else if (arg == "--seed") {
        if (i + 1 >= argc || !ParseU64(argv[++i], &cfg.seed)) {
          std::fprintf(stderr, "Invalid value for --seed\n");
          return 1;
        }
      } else if (arg == "--sample-seed") {
        if (i + 1 >= argc || !ParseU64(argv[++i], &cfg.sample_seed)) {
          std::fprintf(stderr, "Invalid value for --sample-seed\n");
          return 1;
        }
      } else if (arg == "--checkpoint") {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "Missing value for --checkpoint\n");
          return 1;
        }
        cfg.checkpoint_path = argv[++i];
      } else if (arg == "--no-cublas") {
        cfg.use_cublas = false;
      } else if (arg == "--no-tf32") {
        cfg.tf32 = false;
      } else {
        std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
        PrintSampleCharUsage();
        return 1;
      }
    }

    auto result = dlcuda::SampleChar(cfg);
    if (!result.ok()) {
      std::fprintf(stderr, "sample-char failed: %s\n", result.status().message().c_str());
      return 1;
    }
    std::printf("%s\n", result.value().c_str());
    return 0;
  }

  std::fprintf(stderr, "Unknown subcommand: %s\n", command.c_str());
  PrintUsage();
  return 1;
}
