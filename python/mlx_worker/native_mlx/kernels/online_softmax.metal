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

inline float2 online_softmax_push(thread OnlineSoftmax& state, float score) {
  float new_max = max(state.max_score, score);
  float factor = fast::exp(state.max_score - new_max);
  float weight = fast::exp(score - new_max);
  state.max_score = new_max;
  state.sum_exp = state.sum_exp * factor + weight;
  return float2(factor, weight);
}

inline float online_softmax_rescale(float partial_max, float global_max) {
  return partial_max == -INFINITY ? 0.0f : fast::exp(partial_max - global_max);
}

}
