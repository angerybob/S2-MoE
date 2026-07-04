#pragma once

#include "llama-graph.h"

#include <cstdint>
#include <vector>

struct llama_model;

void dflash_validate_config(
        uint32_t block_size,
        uint32_t mask_token_id,
        uint32_t vocab_size,
        uint32_t target_layer_count,
        const std::vector<int> & target_layers);

struct llm_build_dflash_encode : public llm_graph_context {
    llm_build_dflash_encode(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_dflash_decode : public llm_graph_context {
    llm_build_dflash_decode(const llama_model & model, const llm_graph_params & params);
};
