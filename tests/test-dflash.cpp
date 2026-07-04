#include "llama-arch.h"
#include "llama-hparams.h"
#include "llama-model-dflash.h"
#include "llama-model-loader.h"
#include "speculative.h"

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    static_assert(std::is_base_of<llm_graph_context, llm_build_dflash_encode>::value);
    static_assert(std::is_base_of<llm_graph_context, llm_build_dflash_decode>::value);

    require(llm_arch_from_string("dflash") == LLM_ARCH_DFLASH, "DFlash architecture lookup failed");

    const LLM_KV kv(LLM_ARCH_DFLASH);
    require(kv(LLM_KV_DFLASH_BLOCK_SIZE) == "dflash.block_size", "DFlash block-size key mismatch");
    require(kv(LLM_KV_DFLASH_MASK_TOKEN_ID) == "dflash.mask_token_id", "DFlash mask-token key mismatch");
    require(kv(LLM_KV_DFLASH_TARGET_LAYERS) == "dflash.target_layers", "DFlash target-layers key mismatch");
    require(
        kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN) == "dflash.attention.sliding_window_pattern",
        "DFlash sliding-window pattern key mismatch");

    const LLM_TN tn(LLM_ARCH_DFLASH);
    require(std::string(tn(LLM_TENSOR_TOKEN_EMBD, "weight")) == "token_embd.weight", "token embedding name mismatch");
    require(std::string(tn(LLM_TENSOR_FC, "weight")) == "fc.weight", "fc name mismatch");
    require(std::string(tn(LLM_TENSOR_ENC_OUTPUT_NORM, "weight")) == "enc.output_norm.weight", "encoder norm name mismatch");
    require(std::string(tn(LLM_TENSOR_ATTN_Q, "weight", 2)) == "blk.2.attn_q.weight", "attention name mismatch");

    dflash_validate_config(8, 151669, 151936, 48, {2, 13, 24, 35, 46});

    bool rejected = false;
    try {
        dflash_validate_config(1, 151669, 151936, 48, {2, 13, 24, 35, 46});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "block_size < 2 was accepted");

    rejected = false;
    try {
        dflash_validate_config(8, 151936, 151936, 48, {2, 13, 24, 35, 46});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "out-of-range mask token was accepted");

    rejected = false;
    try {
        dflash_validate_config(8, 151669, 151936, 48, {2, 49});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "out-of-range target layer was accepted");

    std::vector<float> interleaved = {
        90.0f, 91.0f, 92.0f, 93.0f,
    };
    const std::vector<float> layer_0 = {
        1.0f, 2.0f,
        3.0f, 4.0f,
    };
    const std::vector<float> layer_1 = {
        5.0f, 6.0f,
        7.0f, 8.0f,
    };

    dflash_append_target_features(
        interleaved,
        {layer_0.data(), layer_1.data()},
        /* n_tokens = */ 2,
        /* n_embd = */ 2);

    const std::vector<float> expected = {
        90.0f, 91.0f, 92.0f, 93.0f,
        1.0f, 2.0f, 5.0f, 6.0f,
        3.0f, 4.0f, 7.0f, 8.0f,
    };
    require(interleaved == expected, "target features were not appended in token/layer/hidden order");

    llama_hparams hparams = {};
    hparams.n_layer = 5;
    dflash_configure_swa(hparams, 2048, {true, true, true, true, true});
    require(hparams.swa_type == LLAMA_SWA_TYPE_STANDARD, "DFlash SWA was not enabled");
    require(hparams.n_swa == 2048, "DFlash sliding-window size mismatch");
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        require(hparams.is_swa(il), "DFlash sliding layer pattern mismatch");
    }

    common_speculative_acceptance_metrics metrics;
    metrics.add(3, 3);
    metrics.add(2, 3);
    metrics.add(0, 3);
    metrics.add(1, 3);
    require(metrics.proposals == 4, "proposal count mismatch");
    require(metrics.accepted_tokens == 6, "accepted-token count mismatch");
    require(metrics.mean_accepted_length() == 1.5, "mean accepted length mismatch");
    require(metrics.position_acceptance(0) == 0.75, "first-position acceptance mismatch");
    require(metrics.position_acceptance(1) == 0.50, "second-position acceptance mismatch");
    require(metrics.position_acceptance(2) == 0.25, "third-position acceptance mismatch");

    std::vector<float> expanded_logits;
    dflash_expand_logits(
        expanded_logits,
        {
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
        },
        {0, 1, 2},
        /* target_vocab_size = */ 5);
    require(expanded_logits.size() == 10, "expanded DFlash logits size mismatch");
    require(
        expanded_logits[0] == 1.0f && expanded_logits[2] == 2.0f && expanded_logits[4] == 3.0f,
        "first DFlash logits row was mapped incorrectly");
    require(
        expanded_logits[5] == 4.0f && expanded_logits[7] == 5.0f && expanded_logits[9] == 6.0f,
        "second DFlash logits row was mapped incorrectly");
    require(std::isinf(expanded_logits[1]) && expanded_logits[1] < 0, "unmapped target logit is not -inf");

    const auto hot_experts = llama_parse_hot_experts_json(
        R"({"gpu_experts": [[0, 1], [1, 3]]})");
    std::vector<llama_ssd_registry_record> registry;
    for (int layer = 0; layer < 2; ++layer) {
        for (int expert = 0; expert < 4; ++expert) {
            registry.push_back({
                "blk." + std::to_string(layer) + ".ffn_gate.00" +
                    std::to_string(expert) + ".weight",
                hot_experts.count({layer, expert}) != 0,
            });
        }
    }
    const auto registry_stats = llama_validate_ssd_registry(registry, hot_experts);
    require(registry_stats.total == 8, "SSD registry total mismatch");
    require(registry_stats.resident == 2, "SSD resident count mismatch");
    require(registry_stats.cold == 6, "SSD cold count mismatch");
    require(registry_stats.requested_hot == 2, "SSD requested-hot count mismatch");
    require(
        llama_validate_ssd_registry({}, {}).total == 0,
        "non-SSD drafter produced registry entries");

    bool invalid_json_rejected = false;
    try {
        llama_parse_hot_experts_json(R"({"gpu_experts": [[0, 1, 2]]})");
    } catch (const std::invalid_argument &) {
        invalid_json_rejected = true;
    }
    require(invalid_json_rejected, "malformed hot-expert JSON was accepted");
}
