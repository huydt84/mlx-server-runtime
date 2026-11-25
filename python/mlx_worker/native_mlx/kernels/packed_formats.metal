#pragma once

#include <metal_stdlib>

using namespace metal;

namespace mlx_air {

enum PackedFormat : int {
  PACKED_HT1 = 0,
  PACKED_HT2 = 1,
  PACKED_HT3 = 2,
  PACKED_RTR1 = 3,
  PACKED_RTR2 = 4,
  PACKED_RTR3 = 5,
  PACKED_RTR4 = 6,
  PACKED_NVFP4 = 7,
  PACKED_LM6 = 8,
  PACKED_LM8 = 9,
};

constexpr constant int PACKED_CHUNK = 16;
constexpr constant int PACKED_LUT_HALVES = 4096;

template <int FORMAT>
struct PackedTraits {
  static constexpr constant bool is_hash = FORMAT <= PACKED_HT3;
  static constexpr constant bool is_trellis = FORMAT <= PACKED_RTR4;
  static constexpr constant bool is_lloyd = FORMAT >= PACKED_LM6;
  static constexpr constant int bits = FORMAT == PACKED_HT1 ? 1
      : FORMAT == PACKED_HT2                               ? 2
      : FORMAT == PACKED_HT3                               ? 3
      : FORMAT <= PACKED_RTR4                              ? FORMAT - PACKED_RTR1 + 1
      : FORMAT == PACKED_NVFP4                             ? 4
      : FORMAT == PACKED_LM6                               ? 6
                                                           : 8;
  static constexpr constant int group =
      FORMAT == PACKED_HT1 || FORMAT == PACKED_HT2 ? 4 : (FORMAT == PACKED_HT3 ? 2 : 1);
  static constexpr constant int lut_entries =
      FORMAT == PACKED_HT3 ? 1024 : (is_hash ? 512 : 4096);
  static constexpr constant uint hash_shift = FORMAT == PACKED_HT3 ? 22u : 23u;
  static constexpr constant int lut_halves = is_hash ? 2048 : 4096;
  static constexpr constant int lut_offset =
      is_hash ? FORMAT * 2048 : 3 * 2048 + (FORMAT - PACKED_RTR1) * 4096;
};

inline uint packed_pair(ushort low, ushort high) {
  return as_type<uint>(ushort2(low, high));
}

template <int FORMAT, int N>
inline void packed_decode_trellis(
    const device ushort* row,
    uint chunk,
    threadgroup const half* lut,
    thread half4* w) {
  using T = PackedTraits<FORMAT>;
  constexpr int bits = T::bits;
  constexpr int group = T::group;
  constexpr uint row_units = uint(N / PACKED_CHUNK * bits);
  uint first = row_units - (chunk + 1u) * uint(bits);
  ushort u[bits + 1];
  for (int i = 0; i < bits; i++) {
    u[i] = row[first + uint(i)];
  }
  uint next = first + uint(bits);
  u[bits] = next < row_units ? row[next] : ushort(0);
#pragma unroll
  for (int j = 0; j < PACKED_CHUNK / group; j++) {
    const int bit = (PACKED_CHUNK - 1 - (j * group + group - 1)) * bits;
    uint window =
        extract_bits(packed_pair(u[bit >> 4], u[(bit >> 4) + 1]), uint(bit & 15), 12u);
    if (T::is_hash) {
      uint index = (window * 0x9E3779B1u) >> T::hash_shift;
      if (group == 4) {
        w[j] = reinterpret_cast<threadgroup const half4*>(lut)[index];
      } else {
        half2 pair = reinterpret_cast<threadgroup const half2*>(lut)[index];
        if (j & 1) {
          w[j >> 1].zw = pair;
        } else {
          w[j >> 1].xy = pair;
        }
      }
    } else {
      w[j >> 2][j & 3] = lut[window];
    }
  }
}

template <int FORMAT>
inline void packed_decode_lloyd(
    const device ushort* row,
    uint chunk,
    threadgroup const float* levels,
    thread float4* w) {
  constexpr int bits = PackedTraits<FORMAT>::bits;
  ushort u[bits + 1];
  for (int i = 0; i < bits; i++) {
    u[i] = row[chunk * uint(bits) + uint(i)];
  }
  u[bits] = 0;
#pragma unroll
  for (int j = 0; j < PACKED_CHUNK; j++) {
    const int bit = j * bits;
    uint index = extract_bits(
        packed_pair(u[bit >> 4], u[(bit >> 4) + 1]), uint(bit & 15), uint(bits));
    w[j >> 2][j & 3] = levels[index];
  }
}

constexpr constant half PACKED_E2M1[16] = {
    0.0h, 0.5h, 1.0h, 1.5h, 2.0h, 3.0h, 4.0h, 6.0h,
    -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h};

inline float packed_e4m3(uchar code) {
  uint magnitude = code & 0x7Fu;
  uint exponent = magnitude >> 3;
  uint mantissa = magnitude & 7u;
  float value = exponent == 0u
      ? float(mantissa) * 0.001953125f
      : as_type<float>(((exponent + 120u) << 23) | (mantissa << 20));
  return (code & 0x80u) ? -value : value;
}

inline void packed_decode_nvfp4(
    const device ushort* row,
    uint chunk,
    thread half4* w) {
#pragma unroll
  for (int i = 0; i < 4; i++) {
    uint code = row[chunk * 4u + uint(i)];
    w[i] = half4(
        PACKED_E2M1[code & 15u],
        PACKED_E2M1[(code >> 4) & 15u],
        PACKED_E2M1[(code >> 8) & 15u],
        PACKED_E2M1[(code >> 12) & 15u]);
  }
}

}
