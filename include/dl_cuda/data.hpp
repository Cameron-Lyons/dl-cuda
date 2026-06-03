#pragma once

#include "dl_cuda/status.hpp"

#include <array>
#include <string>
#include <vector>

namespace dlcuda {

class CharVocab {
public:
  CharVocab() : char_to_id_(256, -1) {}

  static Result<CharVocab> Build(const std::string &text) {
    if (text.empty()) {
      return Status::InvalidArgument("Cannot build vocabulary from empty text");
    }
    std::array<bool, 256> present{};
    for (unsigned char ch : text) {
      present[ch] = true;
    }

    CharVocab vocab;
    vocab.id_to_char_.reserve(256);
    for (size_t i = 0; i < present.size(); ++i) {
      if (present[i]) {
        vocab.id_to_char_.push_back(static_cast<char>(i));
      }
    }
    vocab.char_to_id_.assign(256, -1);
    for (size_t i = 0; i < vocab.id_to_char_.size(); ++i) {
      unsigned char ch = static_cast<unsigned char>(vocab.id_to_char_[i]);
      vocab.char_to_id_[ch] = static_cast<int>(i);
    }
    return vocab;
  }

  [[nodiscard]] int size() const {
    return static_cast<int>(id_to_char_.size());
  }

  [[nodiscard]] int Encode(char ch) const {
    return char_to_id_[static_cast<unsigned char>(ch)];
  }

  [[nodiscard]] char Decode(int id) const {
    return id_to_char_.at(static_cast<size_t>(id));
  }

private:
  std::vector<int> char_to_id_;
  std::vector<char> id_to_char_;
};

} // namespace dlcuda
