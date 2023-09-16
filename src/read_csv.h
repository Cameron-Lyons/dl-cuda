#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

struct Data {
  int a, b;
};

std::vector<Data> readCSV(const std::string &filename) {
  std::ifstream file(filename);
  std::string line;
  std::vector<Data> data;

  std::getline(file, line);

  while (std::getline(file, line)) {
    std::stringstream ss(line);
    Data d;
    std::string value;

    std::getline(ss, value, ',');
    d.a = std::stoi(value);

    std::getline(ss, value, ',');
    d.b = std::stoi(value);

    data.push_back(d);
  }

  return data;
}
