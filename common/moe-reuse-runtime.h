#pragma once

#include <cstdint>

enum class s2moe_reuse_phase {
    prefill,
    draft_decode,
    target_verify,
    ordinary_decode,
};

constexpr bool s2moe_runtime_reuse_enabled(bool requested, s2moe_reuse_phase phase) {
    return requested && phase == s2moe_reuse_phase::target_verify;
}

float s2moe_runtime_reuse_strength(
        const float * logits,
        int32_t n_experts,
        int32_t n_tokens,
        int32_t top_k);
