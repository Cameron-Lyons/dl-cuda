class Operation {
public:
  virtual void forward(float *input, float *output) = 0;
};
