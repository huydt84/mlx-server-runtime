// Online (streaming) softmax helpers shared by attention kernels.
//
// A kernel that sees scores one at a time keeps a running maximum and a
// running sum of exponentials, rescaling what it has accumulated whenever the
// maximum grows. Partial results computed over disjoint key ranges are merged
// by rescaling each to the global maximum.
#pragma once

#include <metal_stdlib>

using namespace metal;

namespace mlx_air {

struct OnlineSoftmax {
  float max_score;
  float sum_exp;
};

inline OnlineSoftmax online_softmax_init() {
  return OnlineSoftmax{-INFINITY, 0.0f};
}

// Adds `score` to the running state. Returns the factor that rescales values
// accumulated so far (x) and the weight of the new score's value (y).
inline float2 online_softmax_push(thread OnlineSoftmax& state, float score) {
  float new_max = max(state.max_score, score);
  float factor = fast::exp(state.max_score - new_max);
  float weight = fast::exp(score - new_max);
  state.max_score = new_max;
  state.sum_exp = state.sum_exp * factor + weight;
  return float2(factor, weight);
}

// Factor that rescales a partial result with maximum `partial_max` to
// `global_max`. An empty partial (maximum -inf) contributes nothing.
inline float online_softmax_rescale(float partial_max, float global_max) {
  return partial_max == -INFINITY ? 0.0f : fast::exp(partial_max - global_max);
}

}  // namespace mlx_air
