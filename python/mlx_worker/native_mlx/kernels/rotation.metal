#pragma once

#include <metal_stdlib>

using namespace metal;

namespace mlx_air {

template <int A, int B, int SIMDGROUPS, typename Load, typename Store>
inline void rotate_vector(
    const thread Load& load,
    const device float* factor,
    const thread Store& store,
    uint simd,
    uint lane) {
  constexpr int LOW = A < 32 ? A : 32;
  constexpr int HIGH = A / LOW;
  constexpr int SUBS = 32 / LOW;
  constexpr int SHARES = SIMDGROUPS * SUBS;
  constexpr int COLUMNS = (B + SHARES - 1) / SHARES;
  const uint row = lane % uint(LOW);
  const uint first = (simd * uint(SUBS) + lane / uint(LOW)) * uint(COLUMNS);
  float acc[HIGH][COLUMNS];
#pragma unroll
  for (int h = 0; h < HIGH; h++) {
#pragma unroll
    for (int c = 0; c < COLUMNS; c++) {
      acc[h][c] = 0.0f;
    }
  }
  for (int k = 0; k < B; k++) {
    float weights[COLUMNS];
#pragma unroll
    for (int c = 0; c < COLUMNS; c++) {
      weights[c] = factor[min(first + uint(c), uint(B - 1)) * uint(B) + uint(k)];
    }
#pragma unroll
    for (int h = 0; h < HIGH; h++) {
      const float value = load((uint(h * LOW) + row) * uint(B) + uint(k));
#pragma unroll
      for (int c = 0; c < COLUMNS; c++) {
        acc[h][c] = fma(value, weights[c], acc[h][c]);
      }
    }
  }
#pragma unroll
  for (int bit = 1; bit < HIGH; bit <<= 1) {
#pragma unroll
    for (int h = 0; h < HIGH; h++) {
      if (h & bit) {
        continue;
      }
#pragma unroll
      for (int c = 0; c < COLUMNS; c++) {
        const float a = acc[h][c];
        const float b = acc[h | bit][c];
        acc[h][c] = a + b;
        acc[h | bit][c] = a - b;
      }
    }
  }
#pragma unroll
  for (int h = 0; h < HIGH; h++) {
#pragma unroll
    for (int c = 0; c < COLUMNS; c++) {
      float value = acc[h][c];
#pragma unroll
      for (int bit = 1; bit < LOW; bit <<= 1) {
        const float other = simd_shuffle_xor(value, ushort(bit));
        value = (row & uint(bit)) ? other - value : value + other;
      }
      if (first + uint(c) < uint(B)) {
        store((uint(h * LOW) + row) * uint(B) + first + uint(c), value);
      }
    }
  }
}

enum RotationInput : int { ROTATE_PLAIN = 0, ROTATE_SIGMOID = 1, ROTATE_SOFT = 2 };

template <int INPUT, int WIDTH, typename X>
struct RotationSource {
  const device X* x;
  const device float* gate;
  const device float* signs;
  float2 caps;

  float raw(uint i) const {
    if (INPUT == ROTATE_SIGMOID) {
      return float(x[i]) / (1.0f + exp(-gate[i]));
    }
    if (INPUT == ROTATE_SOFT) {
      const float g = float(x[i]);
      const float up = float(x[uint(WIDTH) + i]);
      return tanh(g / caps.x) * caps.x / (1.0f + exp(-g)) * (tanh(up / caps.y) * caps.y);
    }
    return float(x[i]);
  }

  float operator()(uint i) const { return raw(i) * signs[i]; }
};

template <typename Y>
struct RotationTarget {
  device Y* y;

  void operator()(uint i, float value) const { y[i] = Y(value); }
};

struct RotationColumns {
  const device float* x;

  float operator()(uint i) const { return x[i]; }
};

struct SignedRotationTarget {
  device float* y;
  const device float* signs;

  void operator()(uint i, float value) const { y[i] = value * signs[i]; }
};

struct RotationStaged {
  threadgroup const float* x;

  float operator()(uint i) const { return x[i]; }
};

template <int INPUT, int A, int B, int SIMDGROUPS, bool RAW, typename X, typename Y>
inline void rotate_input(
    const thread RotationSource<INPUT, A * B, X>& source,
    const device float* factor,
    device Y* y,
    device Y* raw,
    threadgroup float* staged,
    uint matrix,
    uint simd,
    uint lane) {
  if constexpr (INPUT == ROTATE_SOFT) {
    for (uint i = simd * 32 + lane; i < uint(A * B); i += SIMDGROUPS * 32) {
      const float value = source.raw(i);
      staged[i] = value * source.signs[i];
      if (RAW && matrix == 0) {
        raw[i] = Y(value);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    rotate_vector<A, B, SIMDGROUPS>(
        RotationStaged{staged}, factor, RotationTarget<Y>{y}, simd, lane);
  } else {
    rotate_vector<A, B, SIMDGROUPS>(source, factor, RotationTarget<Y>{y}, simd, lane);
    if (RAW && matrix == 0) {
      for (uint i = simd * 32 + lane; i < uint(A * B); i += SIMDGROUPS * 32) {
        raw[i] = Y(source.raw(i));
      }
    }
  }
}

template <int A, int B, int SIMDGROUPS>
inline void rotate_output(
    const device float* x,
    const device float* signs,
    const device float* factor,
    device float* y,
    uint simd,
    uint lane) {
  rotate_vector<A, B, SIMDGROUPS>(
      RotationColumns{x}, factor, SignedRotationTarget{y, signs}, simd, lane);
}

template <int WIDTH, int SIMDGROUPS>
inline void copy_output(const device float* x, device float* y, uint simd, uint lane) {
  for (uint i = simd * 32 + lane; i < uint(WIDTH); i += SIMDGROUPS * 32) {
    y[i] = x[i];
  }
}

}
