#pragma once

namespace mlx_air {

template <int LAGS, int C, typename Index>
inline float causal_input(
    const device float* x,
    ulong stride,
    uint offset,
    const device float* history,
    Index rows,
    Index index,
    uint v,
    uint c,
    int lag) {
  const int at = int(index[v]) - lag;
  if (at >= 0) {
    return x[ulong(v - uint(lag)) * stride + offset + c];
  }
  return history[(ulong(rows[v]) * (LAGS - 1) + uint(LAGS - 1 + at)) * C + c];
}

template <int LAGS, int C, typename Rows>
inline float causal_next_history(
    const device float* x,
    ulong stride,
    uint offset,
    const device float* history,
    Rows starts,
    Rows lengths,
    uint row,
    uint slot,
    uint c) {
  const int at = int(lengths[row]) - (LAGS - 1) + int(slot);
  if (at >= 0) {
    return x[ulong(starts[row] + uint(at)) * stride + offset + c];
  }
  return history[(ulong(row) * (LAGS - 1) + uint(LAGS - 1 + at)) * C + c];
}

template <
    int LAGS,
    int HEADS,
    int HEAD_DIM,
    bool DYNAMIC,
    typename K,
    typename Index,
    typename Dynamic>
inline void causal_keys_values(
    const device float* x,
    uint stride,
    uint offset,
    const device float* history,
    Index rows,
    Index index,
    const device float* weights,
    Dynamic dynamic_weights,
    float norm,
    float clip,
    float eps,
    device K* out,
    uint2 group,
    uint lane) {
  constexpr int C = HEADS * HEAD_DIM;
  constexpr int EPL = HEAD_DIM / 32;
  const uint v = group.y;
  const uint head = group.x;
  float values[EPL];
  float square = 0.0f;
  for (int i = 0; i < EPL; i++) {
    const uint c = head * HEAD_DIM + lane * EPL + i;
    float acc = x[ulong(v) * stride + offset + c];
    for (int lag = 0; lag < LAGS; lag++) {
      float w = weights[lag * C + c];
      if (DYNAMIC) {
        w += dynamic_weights[(v * HEADS + head) * LAGS + lag];
      }
      acc += w *
          causal_input<LAGS, C>(x, stride, offset, history, rows, index, v, c, lag);
    }
    values[i] = acc * norm;
    square += values[i] * values[i];
  }
  square = simd_sum(square);
  const float factor = head < HEADS / 2
      ? min(clip / (sqrt(square) + 1e-6f), 1.0f)
      : rsqrt(square / HEAD_DIM + eps);
  for (int i = 0; i < EPL; i++) {
    out[(ulong(v) * HEADS + head) * HEAD_DIM + lane * EPL + i] = K(values[i] * factor);
  }
}

}
