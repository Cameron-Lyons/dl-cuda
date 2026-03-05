#pragma once

#include "dl_cuda/status.hpp"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace dlcuda {

class CharVocab {
public:
  static Result<CharVocab> Build(const std::string &text) {
    if (text.empty()) {
      return Status::InvalidArgument("Cannot build vocabulary from empty text");
    }
    std::set<char> chars(text.begin(), text.end());
    std::vector<char> id_to_char(chars.begin(), chars.end());
    std::sort(id_to_char.begin(), id_to_char.end());

    CharVocab vocab;
    vocab.id_to_char_ = id_to_char;
    vocab.char_to_id_.assign(256, -1);
    for (size_t i = 0; i < vocab.id_to_char_.size(); ++i) {
      unsigned char ch = static_cast<unsigned char>(vocab.id_to_char_[i]);
      vocab.char_to_id_[ch] = static_cast<int>(i);
    }
    return vocab;
  }

  int size() const { return static_cast<int>(id_to_char_.size()); }

  int Encode(char ch) const {
    return char_to_id_[static_cast<unsigned char>(ch)];
  }

  char Decode(int id) const { return id_to_char_.at(static_cast<size_t>(id)); }

private:
  std::vector<int> char_to_id_;
  std::vector<char> id_to_char_;
};

} // namespace dlcuda
