#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct CSVData {
  std::vector<std::vector<float>> features;
  std::vector<float> labels;
};

CSVData readCSV(const std::string &filename, int label_col = -1) {
  std::ifstream file(filename);
  std::string line;
  CSVData data;

  std::getline(file, line);

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string value;
    std::vector<float> row;

    while (std::getline(ss, value, ',')) {
      row.push_back(std::stof(value));
    }

    if (label_col >= 0 && label_col < static_cast<int>(row.size())) {
      data.labels.push_back(row[label_col]);
      row.erase(row.begin() + label_col);
    }
    data.features.push_back(row);
  }

  return data;
}
