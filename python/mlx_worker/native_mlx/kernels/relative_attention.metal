#pragma once

#include "paged_decode.metal"

namespace mlx_air {

template <
    typename P,
    int QUERY_HEADS,
    int KV_HEADS,
    int HEAD_DIM,
    int PAGE_SIZE,
    int REL_DIM,
    int EXTENT,
    int SIMDGROUPS,
    int SPLITS,
    int MIN_KEYS,
    typename Table,
    typename RowInts,
    typename TokenInts,
    typename Sinks>
inline void relative_paged_attention(
    const device float* queries,
    const device P* pages,
    Table block_tables,
    int table_width,
    RowInts bases,
    TokenInts token_rows,
    TokenInts positions,
    const device float* rel,
    const device float* rel_proj,
    Sinks sinks,
    float scale,
    int window,
    device float* out,
    device float* partial_out,
    device float* partial_max,
    device float* partial_sum,
    threadgroup PagedDecodeScratch<SIMDGROUPS, QUERY_HEADS / KV_HEADS, HEAD_DIM>&
        scratch,
    uint2 group,
    uint simd,
    uint lane,
    uint tid) {
  constexpr int EPL = HEAD_DIM / 32;
  constexpr int GQA = QUERY_HEADS / KV_HEADS;
  const uint kv_head = group.x % KV_HEADS;
  const uint query = group.x / KV_HEADS;
  const uint row = uint(token_rows[query]);
  const uint split = group.y;
  const uint first_head = query * QUERY_HEADS + kv_head * GQA;

  float qv[GQA][EPL];
  float rv[GQA];
  float o[GQA][EPL];
  OnlineSoftmax state[GQA];
  for (int g = 0; g < GQA; g++) {
    const uint head = first_head + g;
    const device float* q = queries + head * HEAD_DIM + lane * EPL;
    for (int i = 0; i < EPL; i++) {
      qv[g][i] = q[i] * scale;
      o[g][i] = 0.0f;
    }
    rv[g] = int(lane) < REL_DIM ? rel[head * REL_DIM + lane] : 0.0f;
    if (simd == 0 && split == 0) {
      state[g] = OnlineSoftmax{sinks[head], 1.0f};
    } else {
      state[g] = online_softmax_init();
    }
  }

  const int position = int(positions[query]);
  const int base = int(bases[row]);
  const int length = position + 1;
  const int start = window > 0 ? max(base, length - window) : base;
  int splits = SPLITS;
  while (splits > 1 && length - start < splits * MIN_KEYS) {
    splits /= 2;
  }
  const int span = (length - start + splits - 1) / splits;
  const int split_start = start + min(int(split), splits) * span;
  const int split_end = min(length, split_start + span);
  auto table = block_tables + row * table_width;
  for (int t = split_start + int(simd); t < split_end; t += SIMDGROUPS) {
    const uint slot = uint(t - base);
    const uint page = uint(table[slot / PAGE_SIZE]);
    const uint offset = (page * PAGE_SIZE + slot % PAGE_SIZE) * 2 * KV_HEADS + kv_head;
    const device P* k_page = pages + ulong(offset) * HEAD_DIM + lane * EPL;
    const device P* v_page = k_page + KV_HEADS * HEAD_DIM;
    float k[EPL];
    float v[EPL];
    for (int i = 0; i < EPL; i++) {
      k[i] = static_cast<float>(k_page[i]);
      v[i] = static_cast<float>(v_page[i]);
    }
    const int distance = position - t;
    const float proj = distance < EXTENT && int(lane) < REL_DIM
        ? rel_proj[distance * REL_DIM + int(lane)]
        : 0.0f;
    for (int g = 0; g < GQA; g++) {
      float score = rv[g] * proj;
      for (int i = 0; i < EPL; i++) {
        score += qv[g][i] * k[i];
      }
      float2 scales = online_softmax_push(state[g], simd_sum(score));
      for (int i = 0; i < EPL; i++) {
        o[g][i] = o[g][i] * scales.x + scales.y * v[i];
      }
    }
  }

  for (int g = 0; g < GQA; g++) {
    if (lane == 0) {
      scratch.maxes[g * SIMDGROUPS + simd] = state[g].max_score;
      scratch.sums[g * SIMDGROUPS + simd] = state[g].sum_exp;
    }
    for (int i = 0; i < EPL; i++) {
      scratch.outputs[(simd * GQA + g) * HEAD_DIM + lane * EPL + i] = o[g][i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint e = tid; e < GQA * HEAD_DIM; e += SIMDGROUPS * 32) {
    uint g = e / HEAD_DIM;
    float group_max = -INFINITY;
    for (int s = 0; s < SIMDGROUPS; s++) {
      group_max = max(group_max, scratch.maxes[g * SIMDGROUPS + s]);
    }
    float total = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < SIMDGROUPS; s++) {
      float factor =
          online_softmax_rescale(scratch.maxes[g * SIMDGROUPS + s], group_max);
      total += scratch.sums[g * SIMDGROUPS + s] * factor;
      acc += scratch.outputs[(s * GQA + g) * HEAD_DIM + (e % HEAD_DIM)] * factor;
    }
    if (SPLITS == 1) {
      out[first_head * HEAD_DIM + e] = total > 0.0f ? acc / total : 0.0f;
    } else {
      uint head = first_head + g;
      partial_out[(head * SPLITS + split) * HEAD_DIM + e % HEAD_DIM] = acc;
      if (e % HEAD_DIM == 0) {
        partial_max[head * SPLITS + split] = group_max;
        partial_sum[head * SPLITS + split] = total;
      }
    }
  }
}

}
