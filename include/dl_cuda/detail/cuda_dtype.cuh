#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>

namespace dlcuda::detail {

__host__ __device__ inline uint32_t FloatToBits(float value) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(value);
#else
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
#endif
}

__host__ __device__ inline float BitsToFloat(uint32_t bits) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(bits);
#else
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

__host__ __device__ inline float Float16BitsToFloat(uint16_t value) {
  uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x03ffu;
  uint32_t bits = sign;

  if (exponent == 0) {
    if (mantissa != 0) {
      int32_t unbiased_exponent = -14;
      while ((mantissa & 0x0400u) == 0) {
        mantissa <<= 1;
        --unbiased_exponent;
      }
      mantissa &= 0x03ffu;
      bits |= static_cast<uint32_t>(unbiased_exponent + 127) << 23;
      bits |= mantissa << 13;
    }
  } else if (exponent == 0x1fu) {
    bits |= 0x7f800000u | (mantissa << 13);
  } else {
    bits |= (exponent + 112u) << 23;
    bits |= mantissa << 13;
  }

  return BitsToFloat(bits);
}

__host__ __device__ inline uint16_t FloatToFloat16Bits(float value) {
  uint32_t bits = FloatToBits(value);
  uint32_t sign = (bits >> 16) & 0x8000u;
  uint32_t abs_bits = bits & 0x7fffffffu;
  uint32_t mantissa = abs_bits & 0x007fffffu;

  if (abs_bits >= 0x7f800000u) {
    uint32_t payload = mantissa >> 13;
    if (payload == 0 && mantissa != 0) {
      payload = 1;
    }
    return static_cast<uint16_t>(sign | 0x7c00u | payload);
  }

  int32_t exponent = static_cast<int32_t>((abs_bits >> 23) & 0xffu) - 127 + 15;
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa |= 0x00800000u;
    int32_t shift = 14 - exponent;
    uint32_t rounded = mantissa >> shift;
    uint32_t remainder_mask = (1u << shift) - 1u;
    uint32_t remainder = mantissa & remainder_mask;
    uint32_t halfway = 1u << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1u) != 0)) {
      ++rounded;
    }
    return static_cast<uint16_t>(sign | rounded);
  }

  uint32_t half_exponent = static_cast<uint32_t>(exponent) << 10;
  uint32_t half_mantissa = mantissa >> 13;
  uint32_t remainder = mantissa & 0x1fffu;
  if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u) != 0)) {
    ++half_mantissa;
    if (half_mantissa == 0x0400u) {
      half_mantissa = 0;
      ++exponent;
      if (exponent >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
      }
      half_exponent = static_cast<uint32_t>(exponent) << 10;
    }
  }

  return static_cast<uint16_t>(sign | half_exponent | half_mantissa);
}

__host__ __device__ inline float BFloat16BitsToFloat(uint16_t value) {
  return BitsToFloat(static_cast<uint32_t>(value) << 16);
}

__host__ __device__ inline uint16_t FloatToBFloat16Bits(float value) {
  uint32_t bits = FloatToBits(value);
  uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

struct Float32Codec {
  using Storage = float;

  __host__ __device__ static float FromStorage(Storage value) {
    return value;
  }

  __host__ __device__ static Storage ToStorage(float value) {
    return value;
  }

  __device__ static float Load(const Storage *data, int64_t index) {
    return data[index];
  }

  __device__ static void Store(Storage *data, int64_t index, float value) {
    data[index] = value;
  }
};

struct Float16Codec {
  using Storage = uint16_t;

  __host__ __device__ static float FromStorage(Storage value) {
    return Float16BitsToFloat(value);
  }

  __host__ __device__ static Storage ToStorage(float value) {
    return FloatToFloat16Bits(value);
  }

  __device__ static float Load(const Storage *data, int64_t index) {
    return FromStorage(data[index]);
  }

  __device__ static void Store(Storage *data, int64_t index, float value) {
    data[index] = ToStorage(value);
  }
};

struct BFloat16Codec {
  using Storage = uint16_t;

  __host__ __device__ static float FromStorage(Storage value) {
    return BFloat16BitsToFloat(value);
  }

  __host__ __device__ static Storage ToStorage(float value) {
    return FloatToBFloat16Bits(value);
  }

  __device__ static float Load(const Storage *data, int64_t index) {
    return FromStorage(data[index]);
  }

  __device__ static void Store(Storage *data, int64_t index, float value) {
    data[index] = ToStorage(value);
  }
};

} // namespace dlcuda::detail
