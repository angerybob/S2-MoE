#include "moe-reuse-runtime.h"

#include <cmath>
#include <cstdio>
#include <vector>

static bool near(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

int main() {
    if (s2moe_runtime_reuse_enabled(true, s2moe_reuse_phase::prefill) ||
        s2moe_runtime_reuse_enabled(true, s2moe_reuse_phase::draft_decode) ||
        s2moe_runtime_reuse_enabled(true, s2moe_reuse_phase::ordinary_decode) ||
        !s2moe_runtime_reuse_enabled(true, s2moe_reuse_phase::target_verify) ||
        s2moe_runtime_reuse_enabled(false, s2moe_reuse_phase::target_verify)) {
        std::fprintf(stderr, "runtime reuse must be scoped to target verification\n");
        return 1;
    }

    // Rows are tokens, columns are experts. For target top-k=2 the margins
    // top1 - top(k+1) are 5-0, 4-0, and 6-1.
    const std::vector<float> logits {
        5.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 4.0f, 3.0f, 0.0f,
        0.0f, 1.0f, 6.0f, 2.0f,
    };

    const float strength = s2moe_runtime_reuse_strength(logits.data(), 4, 3, 2);
    if (!near(strength, 14.0f / 3.0f)) {
        std::fprintf(stderr, "unexpected runtime reuse strength: %.6f\n", strength);
        return 1;
    }

    if (s2moe_runtime_reuse_strength(nullptr, 4, 3, 2) != 0.0f ||
        s2moe_runtime_reuse_strength(logits.data(), 4, 3, 4) != 0.0f) {
        std::fprintf(stderr, "invalid input must disable runtime reuse strength\n");
        return 1;
    }

    return 0;
}
