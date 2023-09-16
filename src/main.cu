#include "layers.cuh"
#include "read_csv.h"
#include "sequential.cuh"
#include "transformer.cuh"
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

int main() {
  std::vector<float *> data;
  readCSV("example.csv", data);

  if (data.empty()) {
    std::cerr << "No data loaded from CSV." << std::endl;
    return 1;
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
