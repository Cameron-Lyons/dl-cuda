#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

void loadCSV(const std::string &filename, std::vector<float *> &data) {
  std::ifstream file(filename);
  std::string line;

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    std::string item;
    std::vector<float> row;

    while (std::getline(ss, item, ',')) {
      row.push_back(std::stof(item));
    }

    float *rowData = new float[row.size()];
    std::copy(row.begin(), row.end(), rowData);
    data.push_back(rowData);
  }
}
