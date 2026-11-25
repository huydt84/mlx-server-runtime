#pragma once

#include "packed_formats.metal"

namespace mlx_air {

constexpr constant int PACKED_DESCRIPTOR = 8;
constexpr constant int PACKED_MAX_TOKENS = 8;
constexpr constant int PACKED_BLOCK = 8;
constexpr constant int PACKED_TILE_ROWS = 16;
constexpr constant int PACKED_TILE_COLUMNS = 32;
constexpr constant int PACKED_TILE_WEIGHTS = PACKED_TILE_ROWS * PACKED_TILE_COLUMNS;
constexpr constant int PACKED_SMALL_UNIT = 4;

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

using PackedBlock = simdgroup_matrix<float, PACKED_BLOCK, PACKED_BLOCK>;

inline uint packed_block_row(uint lane) {
  return ((lane >> 4) & 1u) * 4u + ((lane >> 1) & 3u);
}

inline uint packed_block_column(uint lane) {
  return ((lane >> 3) & 1u) * 4u + (lane & 1u) * 2u;
}

inline thread float2& packed_block_share(thread PackedBlock& block) {
  return reinterpret_cast<thread float2&>(block.thread_elements());
}

template <int FORMAT, int N>
inline void packed_decode_tile(
    const thread PackedGroup& g,
    uint first_row,
    uint first_chunk,
    threadgroup const half* lut,
    threadgroup PackedWeight<FORMAT>* tile,
    uint lane) {
  constexpr uint ROWS = PACKED_TILE_ROWS;
  const uint row = min(first_row + lane % ROWS, g.rows - 1u);
  const uint chunk = first_chunk + lane / ROWS;
  vec<PackedWeight<FORMAT>, 4> w[4];
  packed_decode_chunk<FORMAT, N>(g, row, chunk, lut, w);
  const PackedWeight<FORMAT> scale =
      PackedWeight<FORMAT>(packed_chunk_scale<FORMAT, N>(g, row, chunk));
  threadgroup PackedWeight<FORMAT>* target =
      tile + (lane / ROWS) * uint(PACKED_CHUNK) * ROWS + lane % ROWS;
#pragma unroll
  for (uint c = 0; c < uint(PACKED_CHUNK); c++) {
    const PackedWeight<FORMAT> weight = w[c >> 2][c & 3];
    target[c * ROWS] = FORMAT == PACKED_NVFP4 ? weight * scale : weight;
  }
}

template <
    int FORMAT,
    int N,
    int SIMDGROUPS,
    int TOKEN_BLOCKS,
    typename X,
    typename Y>
inline void packed_rows_blocked(
    const thread PackedGroup& g,
    const device X* x,
    const thread uint* x_rows,
    device Y* y,
    const thread uint* y_offsets,
    uint count,
    threadgroup const half* lut,
    threadgroup float* tiles,
    uint simd,
    uint lane) {
  using W = PackedWeight<FORMAT>;
  constexpr uint ROWS = PACKED_TILE_ROWS;
  constexpr uint ROW_BLOCKS = PACKED_TILE_ROWS / PACKED_BLOCK;
  constexpr uint COLUMN_BLOCKS = PACKED_TILE_COLUMNS / PACKED_BLOCK;
  constexpr uint TILE_CHUNKS = PACKED_TILE_COLUMNS / PACKED_CHUNK;
  threadgroup W* tile =
      reinterpret_cast<threadgroup W*>(tiles + simd * uint(PACKED_TILE_WEIGHTS));
  const uint block_row = packed_block_row(lane);
  const uint block_column = packed_block_column(lane);
  const device vec<X, 2>* inputs[TOKEN_BLOCKS];
#pragma unroll
  for (int t = 0; t < TOKEN_BLOCKS; t++) {
    inputs[t] = reinterpret_cast<const device vec<X, 2>*>(
        x + ulong(x_rows[t]) * N + block_column);
  }
  for (uint first = simd * ROWS; first < g.rows; first += SIMDGROUPS * ROWS) {
    PackedBlock acc[TOKEN_BLOCKS][ROW_BLOCKS];
#pragma unroll
    for (int t = 0; t < TOKEN_BLOCKS; t++) {
#pragma unroll
      for (uint r = 0; r < ROW_BLOCKS; r++) {
        acc[t][r] = PackedBlock(0.0f);
      }
    }
    for (uint chunk = 0; chunk < uint(N / PACKED_CHUNK); chunk += TILE_CHUNKS) {
      packed_decode_tile<FORMAT, N>(g, first, chunk, lut, tile, lane);
      simdgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
      for (uint c = 0; c < COLUMN_BLOCKS; c++) {
        const uint column = c * uint(PACKED_BLOCK);
        PackedBlock weights[ROW_BLOCKS];
#pragma unroll
        for (uint r = 0; r < ROW_BLOCKS; r++) {
          const uint row = r * uint(PACKED_BLOCK) + block_column;
          packed_block_share(weights[r]) = float2(
              *reinterpret_cast<threadgroup const vec<W, 2>*>(
                  tile + (column + block_row) * ROWS + row));
        }
#pragma unroll
        for (int t = 0; t < TOKEN_BLOCKS; t++) {
          PackedBlock input;
          packed_block_share(input) =
              float2(inputs[t][(chunk * uint(PACKED_CHUNK) + column) / 2u]);
#pragma unroll
          for (uint r = 0; r < ROW_BLOCKS; r++) {
            simdgroup_multiply_accumulate(acc[t][r], input, weights[r], acc[t][r]);
          }
        }
      }
      simdgroup_barrier(mem_flags::mem_threadgroup);
    }
#pragma unroll
    for (int t = 0; t < TOKEN_BLOCKS; t++) {
      if (uint(t * PACKED_BLOCK) + block_row >= count) {
        continue;
      }
#pragma unroll
      for (uint r = 0; r < ROW_BLOCKS; r++) {
        const float2 totals = packed_block_share(acc[t][r]);
#pragma unroll
        for (uint i = 0; i < 2; i++) {
          const uint row = first + r * uint(PACKED_BLOCK) + block_column + i;
          if (row < g.rows) {
            y[ulong(y_offsets[t]) + row] =
                Y(totals[i] * packed_row_scale<FORMAT>(g, row));
          }
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
    threadgroup float* tiles,
    uint simd,
    uint lane) {
  packed_load_table<FORMAT, SIMDGROUPS>(luts, levels, lut, simd * 32 + lane);
  if (MAX_TOKENS == 1 || count == 1u) {
    packed_rows_single<FORMAT, N, LANES, SIMDGROUPS>(
        g, x + ulong(x_rows[0]) * N, y + y_offsets[0], lut, simd, lane);
  } else if constexpr (MAX_TOKENS > PACKED_MAX_TOKENS) {
    if (count == 2u) {
      packed_rows_multi<FORMAT, N, LANES, SIMDGROUPS, 2>(
          g, x, x_rows, y, y_offsets, lut, simd, lane);
    } else if (count <= uint(PACKED_SMALL_UNIT)) {
      packed_rows_multi<FORMAT, N, LANES, SIMDGROUPS, PACKED_SMALL_UNIT>(
          g, x, x_rows, y, y_offsets, lut, simd, lane);
    } else if (count <= 8u) {
      packed_rows_blocked<FORMAT, N, SIMDGROUPS, 1>(
          g, x, x_rows, y, y_offsets, count, lut, tiles, simd, lane);
    } else if (count <= 16u) {
      packed_rows_blocked<FORMAT, N, SIMDGROUPS, 2>(
          g, x, x_rows, y, y_offsets, count, lut, tiles, simd, lane);
    } else if (count <= 24u) {
      packed_rows_blocked<FORMAT, N, SIMDGROUPS, 3>(
          g, x, x_rows, y, y_offsets, count, lut, tiles, simd, lane);
    } else {
      packed_rows_blocked<FORMAT, N, SIMDGROUPS, 4>(
          g, x, x_rows, y, y_offsets, count, lut, tiles, simd, lane);
    }
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
          g, luts, levels, x, x_rows, y, y_offsets, count, lut, tiles, simd,  \
          lane);                                                              \
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
    threadgroup float* tiles,
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

inline uint packed_unit_member(uint count, uint entry, uint lane) {
  const uint member = count <= uint(PACKED_SMALL_UNIT)
      ? entry
      : entry * uint(PACKED_BLOCK) + packed_block_row(lane);
  return min(member, count - 1u);
}

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
      blob,
      luts,
      descriptor,
      x,
      x_rows,
      y,
      y_offsets,
      count,
      lut,
      static_cast<threadgroup float*>(nullptr),
      simd,
      lane);
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
      blob,
      luts,
      descriptor,
      x,
      x_rows,
      y,
      y_offsets,
      count,
      lut,
      static_cast<threadgroup float*>(nullptr),
      simd,
      lane);
}

template <
    int FORMATS,
    int N,
    int LANES,
    int SIMDGROUPS,
    int MAX_TOKENS,
    typename X,
    typename Y,
    typename Meta>
inline void packed_dense_blocks(
    const device uint* blob,
    const device half* luts,
    const device uint* descriptors,
    const device X* x,
    Meta meta,
    device Y* y,
    threadgroup half* lut,
    threadgroup float* tiles,
    uint2 group,
    uint simd,
    uint lane) {
  const uint tokens = meta[0];
  const uint total_rows = meta[1];
  const device uint* descriptor = descriptors + group.y * PACKED_DESCRIPTOR;
  const uint first_token = group.x * MAX_TOKENS;
  const uint count = min(uint(MAX_TOKENS), tokens - first_token);
  uint x_rows[MAX_TOKENS / PACKED_BLOCK];
  uint y_offsets[MAX_TOKENS / PACKED_BLOCK];
#pragma unroll
  for (uint i = 0; i < uint(MAX_TOKENS / PACKED_BLOCK); i++) {
    const uint token = first_token + packed_unit_member(count, i, lane);
    x_rows[i] = descriptor[6] * tokens + token;
    y_offsets[i] = token * total_rows + descriptor[5];
  }
  packed_dispatch<FORMATS, N, LANES, SIMDGROUPS, MAX_TOKENS>(
      blob, luts, descriptor, x, x_rows, y, y_offsets, count, lut, tiles, simd, lane);
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
inline void packed_expert_blocks(
    const device uint* blob,
    const device half* luts,
    const device uint* descriptors,
    const device X* x,
    Experts experts,
    Order order,
    Units units,
    uint unit_count,
    Meta meta,
    device Y* y,
    threadgroup half* lut,
    threadgroup float* tiles,
    uint2 group,
    uint simd,
    uint lane) {
  const uint slots = meta[0];
  const uint expert_rows = meta[1];
  const uint groups = meta[2];
  const uint x_div = meta[3];
  const uint start = units[group.x];
  if (start >= slots) {
    return;
  }
  const uint end = group.x + 1u < unit_count ? units[group.x + 1u] : slots;
  const uint count = end - start;
  const uint expert = experts[order[start]];
  const device uint* descriptor =
      descriptors + (expert * groups + group.y) * PACKED_DESCRIPTOR;
  const uint first_row = descriptor[6] * (slots / x_div);
  uint x_rows[MAX_TOKENS / PACKED_BLOCK];
  uint y_offsets[MAX_TOKENS / PACKED_BLOCK];
#pragma unroll
  for (uint i = 0; i < uint(MAX_TOKENS / PACKED_BLOCK); i++) {
    const uint slot = order[start + packed_unit_member(count, i, lane)];
    x_rows[i] = slot / x_div + first_row;
    y_offsets[i] = slot * expert_rows + descriptor[5];
  }
  packed_dispatch<FORMATS, N, LANES, SIMDGROUPS, MAX_TOKENS>(
      blob, luts, descriptor, x, x_rows, y, y_offsets, count, lut, tiles, simd, lane);
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
