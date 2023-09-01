class Operation {
public:
  virtual void forward(float *input, float *output) = 0;
};

class Sequential {
  std::vector<Operation *> operations;

public:
  void add(Operation *op) { operations.push_back(op); }
  void forward(float *input, float *output) {
    float *temp_in = input;
    float *temp_out = output;

    for (auto &op : operations) {
      op->forward(temp_in, temp_out);
      temp_in =
          temp_out; // output of current operation is the input to the next
    }
  }
};
