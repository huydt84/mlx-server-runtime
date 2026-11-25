#pragma once

#include <metal_stdlib>

#include "online_softmax.metal"

using namespace metal;

namespace mlx_air {

template <int SIMDGROUPS, int GQA, int HEAD_DIM>
struct PagedDecodeScratch {
  float maxes[SIMDGROUPS * GQA];
  float sums[SIMDGROUPS * GQA];
  float outputs[SIMDGROUPS * GQA * HEAD_DIM];
};

template <
    typename Q,
    typename KV,
    int QUERY_HEADS,
    int KV_HEADS,
    int HEAD_DIM,
    int PAGE_SIZE,
    int SIMDGROUPS,
    int SPLITS,
    typename Table,
    typename Lengths>
inline void paged_decode_attention(
    const device Q* queries,
    const device KV* key_pages,
    const device KV* value_pages,
    Table block_tables,
    int table_width,
    Lengths lengths,
    float scale,
    int window,
    device Q* out,
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
  uint row = group.x / KV_HEADS;
  uint kv_head = group.x % KV_HEADS;
  uint split = group.y;

  float qv[GQA][EPL];
  float o[GQA][EPL];
  OnlineSoftmax state[GQA];
  for (int g = 0; g < GQA; g++) {
    const device Q* q =
        queries + (row * QUERY_HEADS + kv_head * GQA + g) * HEAD_DIM + lane * EPL;
    for (int i = 0; i < EPL; i++) {
      qv[g][i] = static_cast<float>(q[i]) * scale;
      o[g][i] = 0.0f;
    }
    state[g] = online_softmax_init();
  }

  int length = lengths[row];
  int start = window > 0 ? max(0, length - window) : 0;
  int span = (length - start + SPLITS - 1) / SPLITS;
  int split_start = start + int(split) * span;
  int split_end = min(length, split_start + span);
  auto table = block_tables + row * table_width;
  for (int t = split_start + int(simd); t < split_end; t += SIMDGROUPS) {
    uint page = uint(table[t / PAGE_SIZE]);
    uint base = ((page * PAGE_SIZE + uint(t % PAGE_SIZE)) * KV_HEADS + kv_head) *
            HEAD_DIM +
        lane * EPL;
    float k[EPL];
    float v[EPL];
    for (int i = 0; i < EPL; i++) {
      k[i] = static_cast<float>(key_pages[base + i]);
      v[i] = static_cast<float>(value_pages[base + i]);
    }
    for (int g = 0; g < GQA; g++) {
      float score = 0.0f;
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

  uint first_head = row * QUERY_HEADS + kv_head * GQA;
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
      out[first_head * HEAD_DIM + e] =
          static_cast<Q>(total > 0.0f ? acc / total : 0.0f);
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

template <typename Q, int HEAD_DIM, int SPLITS>
inline void paged_decode_merge(
    const device float* partial_out,
    const device float* partial_max,
    const device float* partial_sum,
    device Q* out,
    uint head,
    uint dim) {
  float global_max = -INFINITY;
  for (int s = 0; s < SPLITS; s++) {
    global_max = max(global_max, partial_max[head * SPLITS + s]);
  }
  float total = 0.0f;
  float acc = 0.0f;
  for (int s = 0; s < SPLITS; s++) {
    float factor =
        online_softmax_rescale(partial_max[head * SPLITS + s], global_max);
    total += partial_sum[head * SPLITS + s] * factor;
    acc += partial_out[(head * SPLITS + s) * HEAD_DIM + dim] * factor;
  }
  out[head * HEAD_DIM + dim] = static_cast<Q>(total > 0.0f ? acc / total : 0.0f);
}

}
