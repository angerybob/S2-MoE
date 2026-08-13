#include "moe-reuse-runtime.h"

#include <algorithm>
#include <functional>
#include <vector>

float s2moe_runtime_reuse_strength(
        const float * logits,
        int32_t n_experts,
        int32_t n_tokens,
        int32_t top_k) {
    if (!logits || n_experts <= 1 || n_tokens <= 0 || top_k <= 0 || top_k >= n_experts) {
        return 0.0f;
    }

    float margin_sum = 0.0f;
    for (int32_t token = 0; token < n_tokens; ++token) {
        const float * begin = logits + static_cast<int64_t>(token) * n_experts;
        std::vector<float> row(begin, begin + n_experts);
        std::nth_element(row.begin(), row.begin() + top_k, row.end(), std::greater<float>());
        const float top1 = *std::max_element(row.begin(), row.end());
        margin_sum += top1 - row[top_k];
    }
    return margin_sum / static_cast<float>(n_tokens);
}
