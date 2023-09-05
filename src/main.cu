#include "layers.cuh"
#include "sequential.cuh"
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

int main() {
  std::vector<float *> data;
  loadCSV("your_data.csv", data);

  if (data.empty()) {
    std::cerr << "No data loaded from CSV." << std::endl;
    return 1; // Error code
  }

  LinearLayer layer(data[0].size(), 10, 5);
  Sequential model;
  model.add(&layer);

  for (auto input : data) {
    float output[5];
    model.forward(input, output, data[0].size() * sizeof(float),
                  5 * sizeof(float));

    for (int i = 0; i < 5; i++) {
      std::cout << output[i] << " ";
    }
    std::cout << std::endl;

    delete[] input;
  }

  return 0;
}
