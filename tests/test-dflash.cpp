#include "llama-arch.h"
#include "llama-model-dflash.h"

#include <cstdlib>
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
}
