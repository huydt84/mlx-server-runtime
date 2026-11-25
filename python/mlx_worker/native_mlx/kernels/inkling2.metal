#pragma once

#include <metal_stdlib>

using namespace metal;

namespace mlx_air {

template <int STATES, int QUERIES, int HIDDEN, int SIMDGROUPS>
struct MixScratch {
  float squares[SIMDGROUPS * STATES];
  float dots[SIMDGROUPS * STATES * QUERIES];
};

template <int STATES, int QUERIES, int HIDDEN, int SIMDGROUPS>
inline void inkling2_mix(
    const thread array<const device float*, STATES>& states,
    const device float* queries,
    float eps,
    device float* out,
    uint tokens,
    threadgroup MixScratch<STATES, QUERIES, HIDDEN, SIMDGROUPS>& scratch,
    uint token,
    uint simd,
    uint lane,
    uint tid) {
  constexpr uint VECTORS = HIDDEN / 4;
  constexpr uint THREADS = SIMDGROUPS * 32;
  const device float4* query =
      reinterpret_cast<const device float4*>(queries);
  for (int s = 0; s < STATES; s++) {
    const device float4* state = reinterpret_cast<const device float4*>(
        states[s] + ulong(token) * HIDDEN);
    float square = 0.0f;
    float product[QUERIES];
    for (int q = 0; q < QUERIES; q++) {
      product[q] = 0.0f;
    }
    for (uint i = tid; i < VECTORS; i += THREADS) {
      const float4 value = state[i];
      square += dot(value, value);
      for (int q = 0; q < QUERIES; q++) {
        product[q] += dot(value, query[uint(q) * VECTORS + i]);
      }
    }
    square = simd_sum(square);
    if (lane == 0) {
      scratch.squares[simd * STATES + s] = square;
    }
    for (int q = 0; q < QUERIES; q++) {
      const float total = simd_sum(product[q]);
      if (lane == 0) {
        scratch.dots[(simd * STATES + s) * QUERIES + q] = total;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float weights[QUERIES][STATES];
  float inverse[STATES];
  for (int s = 0; s < STATES; s++) {
    float square = 0.0f;
    for (int g = 0; g < SIMDGROUPS; g++) {
      square += scratch.squares[g * STATES + s];
    }
    inverse[s] = rsqrt(square / float(HIDDEN) + eps);
  }
  for (int q = 0; q < QUERIES; q++) {
    float scores[STATES];
    float highest = -INFINITY;
    for (int s = 0; s < STATES; s++) {
      float product = 0.0f;
      for (int g = 0; g < SIMDGROUPS; g++) {
        product += scratch.dots[(g * STATES + s) * QUERIES + q];
      }
      scores[s] = product * inverse[s] * rsqrt(float(HIDDEN));
      highest = max(highest, scores[s]);
    }
    float total = 0.0f;
    for (int s = 0; s < STATES; s++) {
      scores[s] = exp(scores[s] - highest);
      total += scores[s];
    }
    for (int s = 0; s < STATES; s++) {
      weights[q][s] = scores[s] / total * inverse[s];
    }
  }
  for (uint i = tid; i < VECTORS; i += THREADS) {
    float4 mixed[QUERIES];
    for (int q = 0; q < QUERIES; q++) {
      mixed[q] = float4(0.0f);
    }
    for (int s = 0; s < STATES; s++) {
      const float4 value = reinterpret_cast<const device float4*>(
          states[s] + ulong(token) * HIDDEN)[i];
      for (int q = 0; q < QUERIES; q++) {
        mixed[q] += weights[q][s] * value;
      }
    }
    for (int q = 0; q < QUERIES; q++) {
      reinterpret_cast<device float4*>(
          out + (ulong(q) * tokens + token) * HIDDEN)[i] = mixed[q];
    }
  }
}

template <int DIM, typename Scale>
inline void inkling2_clip_head(
    const device float* x,
    Scale scale,
    uint scale_start,
    float clip,
    float factor,
    device float* out,
    uint lane) {
  constexpr uint EPL = (DIM + 31) / 32;
  float values[EPL];
  float square = 0.0f;
  for (uint i = 0; i < EPL; i++) {
    const uint element = i * 32 + lane;
    values[i] = element < DIM ? x[element] * scale[scale_start + element] : 0.0f;
    square += values[i] * values[i];
  }
  const float norm = sqrt(simd_sum(square));
  const float shrink = min(clip / (norm + 1e-6f), 1.0f) * factor;
  for (uint i = 0; i < EPL; i++) {
    const uint element = i * 32 + lane;
    if (element < DIM) {
      out[element] = values[i] * shrink;
    }
  }
}

template <int EXPERTS, int TOP, typename Logits, typename Bias>
inline void inkling2_route(
    Logits logits,
    Bias bias,
    float scale,
    device uint* chosen,
    device float* weights,
    uint token) {
  float best[TOP];
  uint index[TOP];
  for (int k = 0; k < TOP; k++) {
    best[k] = -INFINITY;
    index[k] = 0;
  }
  for (uint e = 0; e < EXPERTS; e++) {
    const float score =
        1.0f / (1.0f + exp(-logits[token * EXPERTS + e])) + bias[e];
    if (score > best[TOP - 1]) {
      int k = TOP - 1;
      while (k > 0 && score > best[k - 1]) {
        best[k] = best[k - 1];
        index[k] = index[k - 1];
        k--;
      }
      best[k] = score;
      index[k] = e;
    }
  }
  float total = 0.0f;
  float value[TOP];
  for (int k = 0; k < TOP; k++) {
    value[k] = 1.0f / (1.0f + exp(-logits[token * EXPERTS + index[k]]));
    total += value[k];
  }
  for (int k = 0; k < TOP; k++) {
    chosen[token * TOP + k] = index[k];
    weights[token * TOP + k] = value[k] / (total + 1e-20f) * scale;
  }
}


template <int HIDDEN, int RANK, int PIECES, int SIMDGROUPS, typename G>
inline void inkling2_gate_down(
    const device float* hidden,
    const device float* message,
    const device G* down,
    device G* low,
    uint piece,
    uint simd,
    uint lane) {
  constexpr uint SPAN = HIDDEN / 32;
  constexpr uint COLUMNS = RANK / (PIECES * SIMDGROUPS);
  const uint begin = lane * SPAN;
  float2 squares = 0.0f;
  for (uint i = begin; i < begin + SPAN; i++) {
    const float2 value = float2(hidden[i], message[i]);
    squares += value * value;
  }
  const float2 scale = rsqrt(simd_sum(squares) / float(HIDDEN) + 1e-6f);
  const uint first = (piece * SIMDGROUPS + simd) * COLUMNS;
  float acc[COLUMNS] = {0.0f};
  for (uint i = begin; i < begin + SPAN; i++) {
    const float a = float(G(hidden[i] * scale.x));
    const float b = float(G(message[i] * scale.y));
    const device G* row_a = down + i * RANK + first;
    const device G* row_b = down + (i + HIDDEN) * RANK + first;
#pragma unroll
    for (uint c = 0; c < COLUMNS; c++) {
      acc[c] += a * float(row_a[c]) + b * float(row_b[c]);
    }
  }
#pragma unroll
  for (uint c = 0; c < COLUMNS; c++) {
    const float total = simd_sum(acc[c]);
    if (lane == 0) {
      low[first + c] = G(total);
    }
  }
}

template <int HIDDEN, int RANK, int PIECES, int SIMDGROUPS, typename G>
inline void inkling2_gate_up(
    const device float* hidden,
    const device float* message,
    const device G* low,
    const device G* up,
    device float* out,
    uint piece,
    uint tid) {
  constexpr uint THREADS = SIMDGROUPS * 32;
  constexpr uint SHARE = HIDDEN / (PIECES * THREADS);
  const uint start = (piece * THREADS + tid) * SHARE;
  float forget[SHARE] = {0.0f};
  float keep[SHARE] = {0.0f};
  for (uint r = 0; r < RANK; r++) {
    const float value = float(low[r]);
    const device G* row = up + r * 2 * HIDDEN + start;
#pragma unroll
    for (uint i = 0; i < SHARE; i++) {
      forget[i] += value * float(row[i]);
      keep[i] += value * float(row[HIDDEN + i]);
    }
  }
#pragma unroll
  for (uint i = 0; i < SHARE; i++) {
    const uint index = start + i;
    out[index] = (1.0f - tanh(float(G(forget[i])))) * hidden[index] +
        tanh(float(G(keep[i]))) * message[index];
  }
}

template <int WIDTH, int GROUP>
inline void inkling2_row_products(
    const device float* x,
    const device float* weight,
    device float* y,
    uint first,
    uint rows,
    uint lane) {
  constexpr uint SPAN = WIDTH / 32;
  if (first >= rows) {
    return;
  }
  vec<float, GROUP> acc = 0.0f;
  for (uint i = lane * SPAN; i < (lane + 1) * SPAN; i++) {
    const float value = x[i];
#pragma unroll
    for (uint r = 0; r < GROUP; r++) {
      acc[r] += value * weight[min(first + r, rows - 1) * WIDTH + i];
    }
  }
  acc = simd_sum(acc);
  if (lane < GROUP && first + lane < rows) {
    y[first + lane] = acc[lane];
  }
}

}
