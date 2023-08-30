#include <cuda_runtime.h>
#include <vector>

struct Tensor
{
    float *data;
    std::vector<int> shape;
    std::vector<int> strides;

    void calculateStrides()
    {
        strides.resize(shape.size());
        strides[shape.size() - 1] = 1;
        for (int i = shape.size() - 2; i >= 0; --i)
        {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    int getLinearIndex(const std::vector<int> &indices) const
    {
        int index = 0;
        for (size_t i = 0; i < shape.size(); ++i)
        {
            index += indices[i] * strides[i];
        }
        return index;
    }
};