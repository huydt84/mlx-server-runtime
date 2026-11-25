#pragma once

#include "packed_formats.metal"

namespace mlx_air {

constexpr constant int PACKED_DESCRIPTOR = 8;
constexpr constant int PACKED_MAX_TOKENS = 8;

template <int LANES>
inline float packed_sub_sum(float value) {
  if (LANES >= 16) {
    value += simd_shuffle_xor(value, 8u);
  }
  if (LANES >= 8) {
    value += simd_shuffle_xor(value, 4u);
  }
  if (LANES >= 4) {
    value += simd_shuffle_xor(value, 2u);
  }
  if (LANES >= 2) {
    value += simd_shuffle_xor(value, 1u);
  }
  return value;
}

struct PackedGroup {
  const device uint* codes;
  const device half* scales;
  const device uchar* block_scales;
  uint rows;
  float outer;
};

inline PackedGroup packed_group(const device uint* blob, const device uint* descriptor) {
  PackedGroup g;
  g.codes = blob + descriptor[1];
  g.scales = reinterpret_cast<const device half*>(blob) + descriptor[2];
  g.block_scales = reinterpret_cast<const device uchar*>(blob + descriptor[3]);
  g.rows = descriptor[4];
  g.outer = as_type<float>(descriptor[7]);
  return g;
}

template <int FORMAT>
using PackedWeight =
    metal::conditional_t<PackedTraits<FORMAT>::is_lloyd, float, half>;

template <int FORMAT, int N>
inline void packed_decode_chunk(
    const thread PackedGroup& g,
    uint row,
    uint chunk,
    threadgroup const half* lut,
    thread vec<PackedWeight<FORMAT>, 4>* w) {
  constexpr uint words = uint(N * PackedTraits<FORMAT>::bits / 32);
  const device ushort* codes =
      reinterpret_cast<const device ushort*>(g.codes + row * words);
  if constexpr (PackedTraits<FORMAT>::is_trellis) {
    packed_decode_trellis<FORMAT, N>(codes, chunk, lut, w);
  } else if constexpr (FORMAT == PACKED_NVFP4) {
    packed_decode_nvfp4(codes, chunk, w);
  } else {
    packed_decode_lloyd<FORMAT>(
        codes, chunk, reinterpret_cast<threadgroup const float*>(lut), w);
  }
}

template <int FORMAT, int N>
inline float packed_chunk_scale(const thread PackedGroup& g, uint row, uint chunk) {
  if (FORMAT == PACKED_NVFP4) {
    return packed_e4m3(g.block_scales[row * uint(N / PACKED_CHUNK) + chunk]);
  }
  return 1.0f;
}

template <int FORMAT>
inline float packed_row_scale(const thread PackedGroup& g, uint row) {
  return FORMAT == PACKED_NVFP4 ? g.outer : float(g.scales[row]);
}

template <typename W, typename X>
inline float packed_dot(const thread W* w, const thread X* x) {
  return dot(float4(w[0]), float4(x[0])) + dot(float4(w[1]), float4(x[1])) +
      dot(float4(w[2]), float4(x[2])) + dot(float4(w[3]), float4(x[3]));
}

template <int FORMAT, int N, int LANES, int SIMDGROUPS, typename X, typename Y>
inline void packed_rows_single(
    const thread PackedGroup& g,
    const device X* x,
    device Y* y,
    threadgroup const half* lut,
    uint simd,
    uint lane) {
  constexpr uint ROW_BLOCK = 4;
  constexpr uint SUBS = 32 / LANES;
  constexpr uint ITERS = N / (PACKED_CHUNK * LANES);
  const uint sub = lane / LANES;
  const uint column = lane % LANES;
  for (uint block = simd * SUBS + sub; block * ROW_BLOCK < g.rows;
       block += SIMDGROUPS * SUBS) {
    const uint first = block * ROW_BLOCK;
    float acc[ROW_BLOCK] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint it = 0; it < ITERS; it++) {
      const uint chunk = column + LANES * it;
      const device vec<X, 4>* source =
          reinterpret_cast<const device vec<X, 4>*>(x + chunk * PACKED_CHUNK);
      vec<X, 4> xs[4] = {source[0], source[1], source[2], source[3]};
#pragma unroll
      for (uint r = 0; r < ROW_BLOCK; r++) {
        const uint row = min(first + r, g.rows - 1u);
        vec<PackedWeight<FORMAT>, 4> w[4];
        packed_decode_chunk<FORMAT, N>(g, row, chunk, lut, w);
        acc[r] += packed_dot(w, xs) * packed_chunk_scale<FORMAT, N>(g, row, chunk);
      }
    }
#pragma unroll
    for (uint r = 0; r < ROW_BLOCK; r++) {
      float total = packed_sub_sum<LANES>(acc[r]);
      if (column == 0 && first + r < g.rows) {
        y[first + r] = Y(total * packed_row_scale<FORMAT>(g, first + r));
      }
    }
  }
}

template <
    int FORMAT,
    int N,
    int LANES,
    int SIMDGROUPS,
    int COUNT,
    typename X,
    typename Y>
inline void packed_rows_multi(
    const thread PackedGroup& g,
    const device X* x,
    const thread uint* x_rows,
    device Y* y,
    const thread uint* y_offsets,
    threadgroup const half* lut,
    uint simd,
    uint lane) {
  constexpr uint ROW_BLOCK = 2;
  constexpr uint SUBS = 32 / LANES;
  constexpr uint ITERS = N / (PACKED_CHUNK * LANES);
  const uint sub = lane / LANES;
  const uint column = lane % LANES;
  const device X* xs[COUNT];
  for (int t = 0; t < COUNT; t++) {
    xs[t] = x + ulong(x_rows[t]) * N;
  }
  for (uint block = simd * SUBS + sub; block * ROW_BLOCK < g.rows;
       block += SIMDGROUPS * SUBS) {
    const uint first = block * ROW_BLOCK;
    float acc[ROW_BLOCK][COUNT];
    for (uint r = 0; r < ROW_BLOCK; r++) {
      for (int t = 0; t < COUNT; t++) {
        acc[r][t] = 0.0f;
      }
    }
    for (uint it = 0; it < ITERS; it++) {
      const uint chunk = column + LANES * it;
      vec<PackedWeight<FORMAT>, 4> w[ROW_BLOCK][4];
      float block_scale[ROW_BLOCK];
#pragma unroll
      for (uint r = 0; r < ROW_BLOCK; r++) {
        const uint row = min(first + r, g.rows - 1u);
        packed_decode_chunk<FORMAT, N>(g, row, chunk, lut, w[r]);
        block_scale[r] = packed_chunk_scale<FORMAT, N>(g, row, chunk);
      }
#pragma unroll
      for (int t = 0; t < COUNT; t++) {
        const device vec<X, 4>* source =
            reinterpret_cast<const device vec<X, 4>*>(xs[t] + chunk * PACKED_CHUNK);
        vec<X, 4> values[4] = {source[0], source[1], source[2], source[3]};
#pragma unroll
        for (uint r = 0; r < ROW_BLOCK; r++) {
          acc[r][t] += packed_dot(w[r], values) * block_scale[r];
        }
      }
    }
#pragma unroll
    for (uint r = 0; r < ROW_BLOCK; r++) {
      const uint row = first + r;
      const float scale = packed_row_scale<FORMAT>(g, min(row, g.rows - 1u));
#pragma unroll
      for (int t = 0; t < COUNT; t++) {
        float total = packed_sub_sum<LANES>(acc[r][t]);
        if (column == 0 && row < g.rows) {
          y[ulong(y_offsets[t]) + row] = Y(total * scale);
        }
      }
    }
  }
}

template <int FORMAT, int SIMDGROUPS>
inline void packed_load_table(
    const device half* luts,
    const device float* levels,
    threadgroup half* lut,
    uint tid) {
  using T = PackedTraits<FORMAT>;
  if (T::is_trellis) {
    for (uint i = tid; i < uint(T::lut_halves); i += SIMDGROUPS * 32) {
      lut[i] = luts[uint(T::lut_offset) + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else if (T::is_lloyd) {
    threadgroup float* table = reinterpret_cast<threadgroup float*>(lut);
    for (uint i = tid; i < uint(1 << T::bits); i += SIMDGROUPS * 32) {
      table[i] = levels[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <
    int FORMAT,
    int N,
    int LANES,
    int SIMDGROUPS,
    int MAX_TOKENS,
    typename X,
    typename Y>
inline void packed_run_group(
    const thread PackedGroup& g,
    const device half* luts,
    const device float* levels,
    const device X* x,
    const thread uint* x_rows,
    device Y* y,
    const thread uint* y_offsets,
    uint count,
    threadgroup half* lut,
    uint simd,
    uint lane) {
  packed_load_table<FORMAT, SIMDGROUPS>(luts, levels, lut, simd * 32 + lane);
  if (MAX_TOKENS == 1 || count == 1u) {
    packed_rows_single<FORMAT, N, LANES, SIMDGROUPS>(
        g, x + ulong(x_rows[0]) * N, y + y_offsets[0], lut, simd, lane);
  } else if (MAX_TOKENS < PACKED_MAX_TOKENS) {
    packed_rows_multi<FORMAT, N, LANES, SIMDGROUPS, MAX_TOKENS>(
        g, x, x_rows, y, y_offsets, lut, simd, lane);
  } else if (count == 2u) {
    packed_rows_multi<FORMAT, N, LANES, SIMDGROUPS, 2>(
        g, x, x_rows, y, y_offsets, lut, simd, lane);
  } else if (count <= 4u) {
    packed_rows_multi<FORMAT, N, LANES, SIMDGROUPS, 4>(
        g, x, x_rows, y, y_offsets, lut, simd, lane);
  } else {
    packed_rows_multi<FORMAT, N, LANES, SIMDGROUPS, 8>(
        g, x, x_rows, y, y_offsets, lut, simd, lane);
  }
}

#define MLX_AIR_PACKED_CASE(FORMAT)                                           \
  case FORMAT:                                                                \
    if ((FORMATS & (1 << FORMAT)) != 0) {                                     \
      packed_run_group<FORMAT, N, LANES, SIMDGROUPS, MAX_TOKENS>(             \
          g, luts, levels, x, x_rows, y, y_offsets, count, lut, simd, lane);  \
    }                                                                         \
    break;

template <
    int FORMATS,
    int N,
    int LANES,
    int SIMDGROUPS,
    int MAX_TOKENS,
    typename X,
    typename Y>
inline void packed_dispatch(
    const device uint* blob,
    const device half* luts,
    const device uint* descriptor,
    const device X* x,
    const thread uint* x_rows,
    device Y* y,
    const thread uint* y_offsets,
    uint count,
    threadgroup half* lut,
    uint simd,
    uint lane) {
  const PackedGroup g = packed_group(blob, descriptor);
  const device float* levels = reinterpret_cast<const device float*>(blob + descriptor[3]);
  switch (int(descriptor[0])) {
    MLX_AIR_PACKED_CASE(PACKED_HT1)
    MLX_AIR_PACKED_CASE(PACKED_HT2)
    MLX_AIR_PACKED_CASE(PACKED_HT3)
    MLX_AIR_PACKED_CASE(PACKED_RTR1)
    MLX_AIR_PACKED_CASE(PACKED_RTR2)
    MLX_AIR_PACKED_CASE(PACKED_RTR3)
    MLX_AIR_PACKED_CASE(PACKED_RTR4)
    MLX_AIR_PACKED_CASE(PACKED_NVFP4)
    MLX_AIR_PACKED_CASE(PACKED_LM6)
    MLX_AIR_PACKED_CASE(PACKED_LM8)
    default:
      break;
  }
}

#undef MLX_AIR_PACKED_CASE

template <
    int FORMATS,
    int N,
    int LANES,
    int SIMDGROUPS,
    int MAX_TOKENS,
    typename X,
    typename Y,
    typename Meta>
inline void packed_dense_matmul(
    const device uint* blob,
    const device half* luts,
    const device uint* descriptors,
    const device X* x,
    Meta meta,
    device Y* y,
    threadgroup half* lut,
    uint2 group,
    uint simd,
    uint lane) {
  const uint tokens = meta[0];
  const uint total_rows = meta[1];
  const device uint* descriptor = descriptors + group.y * PACKED_DESCRIPTOR;
  const uint first_token = group.x * MAX_TOKENS;
  const uint count = min(uint(MAX_TOKENS), tokens - first_token);
  uint x_rows[8];
  uint y_offsets[8];
  for (uint t = 0; t < 8; t++) {
    const uint token = first_token + min(t, count - 1u);
    x_rows[t] = descriptor[6] * tokens + token;
    y_offsets[t] = token * total_rows + descriptor[5];
  }
  packed_dispatch<FORMATS, N, LANES, SIMDGROUPS, MAX_TOKENS>(
      blob, luts, descriptor, x, x_rows, y, y_offsets, count, lut, simd, lane);
}

template <
    int FORMATS,
    int N,
    int LANES,
    int SIMDGROUPS,
    int MAX_TOKENS,
    typename X,
    typename Y,
    typename Experts,
    typename Order,
    typename Units,
    typename Meta>
inline void packed_expert_matmul(
    const device uint* blob,
    const device half* luts,
    const device uint* descriptors,
    const device X* x,
    Experts experts,
    Order order,
    Units units,
    Meta meta,
    device Y* y,
    threadgroup half* lut,
    uint2 group,
    uint simd,
    uint lane) {
  const uint slots = meta[0];
  const uint expert_rows = meta[1];
  const uint groups = meta[2];
  const uint x_div = meta[3];
  const uint start = MAX_TOKENS == 1 ? group.x : units[group.x];
  if (start >= slots) {
    return;
  }
  const uint leader = MAX_TOKENS == 1 ? start : order[start];
  const uint expert = experts[leader];
  const device uint* descriptor =
      descriptors + (expert * groups + group.y) * PACKED_DESCRIPTOR;
  const uint first_row = descriptor[6] * (slots / x_div);
  uint x_rows[8];
  uint y_offsets[8];
  uint count = 0;
  for (uint t = 0; t < 8; t++) {
    uint slot = leader;
    if (MAX_TOKENS > 1 && t > 0 && start + t < slots &&
        experts[order[start + t]] == expert) {
      slot = order[start + t];
      count = t + 1;
    } else if (t == 0) {
      count = 1;
    }
    x_rows[t] = slot / x_div + first_row;
    y_offsets[t] = slot * expert_rows + descriptor[5];
  }
  packed_dispatch<FORMATS, N, LANES, SIMDGROUPS, MAX_TOKENS>(
      blob, luts, descriptor, x, x_rows, y, y_offsets, count, lut, simd, lane);
}

template <int FORMAT, int N, int SIMDGROUPS>
inline void packed_decode_rows(
    const thread PackedGroup& g,
    uint first,
    uint rows,
    const device half* luts,
    const device float* levels,
    device float* out,
    threadgroup half* lut,
    uint tid) {
  constexpr uint CHUNKS = uint(N / PACKED_CHUNK);
  packed_load_table<FORMAT, SIMDGROUPS>(luts, levels, lut, tid);
  for (uint i = tid; i < rows * CHUNKS; i += SIMDGROUPS * 32) {
    const uint row = first + i / CHUNKS;
    const uint chunk = i % CHUNKS;
    vec<PackedWeight<FORMAT>, 4> w[4];
    packed_decode_chunk<FORMAT, N>(g, row, chunk, lut, w);
    const float scale = packed_row_scale<FORMAT>(g, row) *
        packed_chunk_scale<FORMAT, N>(g, row, chunk);
    device float4* target = reinterpret_cast<device float4*>(out + i * PACKED_CHUNK);
    for (int i = 0; i < 4; i++) {
      target[i] = float4(w[i]) * scale;
    }
  }
}

#define MLX_AIR_PACKED_ROW_CASE(FORMAT)                                       \
  case FORMAT:                                                                \
    if ((FORMATS & (1 << FORMAT)) != 0) {                                     \
      packed_decode_rows<FORMAT, N, SIMDGROUPS>(                              \
          g, first, rows, luts, levels, out, lut, tid);                       \
    }                                                                         \
    break;

template <int FORMATS, int N, int SIMDGROUPS>
inline void packed_decode(
    const device uint* blob,
    const device half* luts,
    const device uint* descriptor,
    uint first,
    uint rows,
    device float* out,
    threadgroup half* lut,
    uint tid) {
  const PackedGroup g = packed_group(blob, descriptor);
  const device float* levels = reinterpret_cast<const device float*>(blob + descriptor[3]);
  switch (int(descriptor[0])) {
    MLX_AIR_PACKED_ROW_CASE(PACKED_HT1)
    MLX_AIR_PACKED_ROW_CASE(PACKED_HT2)
    MLX_AIR_PACKED_ROW_CASE(PACKED_HT3)
    MLX_AIR_PACKED_ROW_CASE(PACKED_RTR1)
    MLX_AIR_PACKED_ROW_CASE(PACKED_RTR2)
    MLX_AIR_PACKED_ROW_CASE(PACKED_RTR3)
    MLX_AIR_PACKED_ROW_CASE(PACKED_RTR4)
    MLX_AIR_PACKED_ROW_CASE(PACKED_NVFP4)
    MLX_AIR_PACKED_ROW_CASE(PACKED_LM6)
    MLX_AIR_PACKED_ROW_CASE(PACKED_LM8)
    default:
      break;
  }
}

template <int FORMATS, int N, int SIMDGROUPS, int GROUP_ROWS, typename Rows>
inline void packed_row_lookup(
    const device uint* blob,
    const device half* luts,
    const device uint* descriptors,
    Rows rows,
    device float* y,
    threadgroup half* lut,
    uint index,
    uint tid) {
  const uint row = uint(rows[index]);
  packed_decode<FORMATS, N, SIMDGROUPS>(
      blob,
      luts,
      descriptors + (row / GROUP_ROWS) * PACKED_DESCRIPTOR,
      row % GROUP_ROWS,
      1,
      y + ulong(index) * N,
      lut,
      tid);
}

template <int FORMATS, int N, int SIMDGROUPS>
inline void packed_decode_bank(
    const device uint* blob,
    const device half* luts,
    const device uint* descriptors,
    device float* y,
    threadgroup half* lut,
    uint index,
    uint tid) {
  const device uint* descriptor = descriptors + index * PACKED_DESCRIPTOR;
  packed_decode<FORMATS, N, SIMDGROUPS>(
      blob, luts, descriptor, 0, descriptor[4], y + ulong(descriptor[5]) * N, lut, tid);
}

#undef MLX_AIR_PACKED_ROW_CASE

template <int EXPERTS, typename Slots>
inline void packed_plan_experts(
    Slots experts, device uint* order, device uint* units, uint slots, uint slot) {
  constexpr uint WORDS = (EXPERTS + 31) / 32;
  uint seen[WORDS];
  for (uint word = 0; word < WORDS; word++) {
    seen[word] = 0u;
  }
  const uint expert = experts[slot];
  uint rank = 0;
  bool leads = true;
  for (uint other = 0; other < slots; other++) {
    const uint value = experts[other];
    const bool same_before = value == expert && other < slot;
    rank += uint(value < expert || same_before);
    leads = leads && !same_before;
    seen[value / 32u] |= 1u << (value % 32u);
  }
  uint below = 0;
  uint distinct = 0;
  for (uint word = 0; word < WORDS; word++) {
    distinct += popcount(seen[word]);
    const uint bits = word < expert / 32u ? seen[word]
        : word == expert / 32u            ? seen[word] & ((1u << (expert % 32u)) - 1u)
                                          : 0u;
    below += popcount(bits);
  }
  order[rank] = slot;
  if (leads) {
    units[below] = rank;
  }
  if (slot >= distinct) {
    units[slot] = slots;
  }
}

}
