#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <vector>
#include <numeric>
#include <cmath>

static std::vector<int> topk_ids(const std::vector<float> & row, int k) {
    std::vector<int> idx(row.size());
    for (size_t i = 0; i < row.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
        if (row[a] == row[b]) return a < b;
        return row[a] > row[b];
    });
    idx.resize(k);
    return idx;
}

struct reuse_stats {
    std::vector<float> w_scaled;
    std::vector<float> g;
    std::vector<int>   global_experts;
};

static reuse_stats compute_importance(
        const std::vector<float> & logits,
        int n_tokens,
        int n_experts,
        int top_k,
        int cap) {
    std::vector<float> w_raw(n_tokens, 0.0f);
    float max_w = 0.0f;
    for (int b = 0; b < n_tokens; ++b) {
        float sum = 0.0f;
        float mx  = -INFINITY;
        for (int e = 0; e < n_experts; ++e) {
            const float v = logits[b * n_experts + e];
            sum += v;
            mx = std::max(mx, v);
        }
        const float mean = sum / (float) n_experts;
        const float w = std::max(0.0f, mx - mean);
        w_raw[b] = w;
        max_w = std::max(max_w, w);
    }

    reuse_stats stats;
    stats.w_scaled.resize(n_tokens, 0.0f);
    if (max_w > 0.0f) {
        const float scale = 1.0f / max_w;
        for (int b = 0; b < n_tokens; ++b) {
            stats.w_scaled[b] = w_raw[b] * scale;
        }
    } else {
        stats.g.assign(n_experts, 0.0f);
        return stats;
    }

    stats.g.assign(n_experts, 0.0f);
    for (int b = 0; b < n_tokens; ++b) {
        const float w = stats.w_scaled[b];
        for (int e = 0; e < n_experts; ++e) {
            stats.g[e] += w * logits[b * n_experts + e];
        }
    }

    int k_global = std::min(n_experts, cap);
    if (k_global < top_k) {
        k_global = top_k;
    }
    std::vector<int> idx(n_experts);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + k_global, idx.end(), [&](int a, int b) {
        if (stats.g[a] == stats.g[b]) {
            return a < b;
        }
        return stats.g[a] > stats.g[b];
    });
    stats.global_experts.assign(idx.begin(), idx.begin() + k_global);
    return stats;
}

int main() {
    const int n_experts = 8;
    const int n_tokens  = 4;
    const int top_k     = 2;
    const float lambda  = 3.0f;
    const int cap       = 4;

    // logits layout: [E, B]
    std::vector<float> logits = {
        // token 0
        1.2f, 0.1f, 0.3f, -0.2f, 0.0f, 0.4f, 0.2f, -0.1f,
        // token 1
        0.9f, 0.8f, 0.7f, 0.6f, 0.2f, 0.3f, 0.1f, 0.0f,
        // token 2
        0.6f, 0.7f, 0.3f, 0.4f, 0.2f, 0.3f, 0.2f, 0.1f,
        // token 3
        0.3f, 0.2f, 0.1f, 0.0f, 0.9f, 0.8f, 0.7f, 0.6f,
    };

    struct ggml_init_params params = {
        /*.mem_size   =*/ 1024*1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    ggml_context * ctx = ggml_init(params);

    int64_t ne[2] = { n_experts, n_tokens };
    ggml_tensor * t_logits = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
    memcpy(t_logits->data, logits.data(), logits.size()*sizeof(float));

    ggml_tensor * biased = ggml_moe_reuse_two_pass(ctx, t_logits, top_k, lambda, cap);
    ggml_tensor * biased_noop = ggml_moe_reuse_two_pass(ctx, t_logits, top_k, 0.0f, 0);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, biased);
    ggml_build_forward_expand(gf, biased_noop);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const float * biased_data = (const float *) biased->data;
    const float * biased_noop_data = (const float *) biased_noop->data;

    const auto stats = compute_importance(logits, n_tokens, n_experts, top_k, cap);

    printf("Token importance (scaled, max=1):\n");
    for (int b = 0; b < n_tokens; ++b) {
        printf("  token %d: %.4f\n", b, stats.w_scaled[b]);
    }

    printf("Global expert scores g[e]:\n  ");
    for (float v : stats.g) printf("%.4f ", v);
    printf("\nGlobal expert set (cap %d): ", cap);
    for (int id : stats.global_experts) printf("%d ", id);
    printf("\n\n");

    printf("Original top-%d per token:\n", top_k);
    for (int b = 0; b < n_tokens; ++b) {
        std::vector<float> row(logits.begin() + b*n_experts, logits.begin() + (b+1)*n_experts);
        auto ids = topk_ids(row, top_k);
        printf("  token %d: ", b);
        for (int id : ids) printf("%d ", id);
        printf("\n");
    }

    printf("Biased logits (reuse reward) and top-%d per token:\n", top_k);
    for (int b = 0; b < n_tokens; ++b) {
        std::vector<float> row(biased_data + b*n_experts, biased_data + (b+1)*n_experts);
        auto ids = topk_ids(row, top_k);
        printf("  token %d logits:", b);
        for (float v : row) printf(" %.3f", v);
        printf(" | topk: ");
        for (int id : ids) printf("%d ", id);
        printf("\n");
    }

    bool noop_matches = true;
    for (size_t i = 0; i < logits.size(); ++i) {
        if (logits[i] != biased_noop_data[i]) {
            noop_matches = false;
            break;
        }
    }
    printf("\nNo-op path (lambda=0, cap=0) keeps logits unchanged: %s\n",
           noop_matches ? "yes" : "no");

    ggml_free(ctx);
    return 0;
}
