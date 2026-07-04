#pragma once

#include "llama-graph.h"

#include <cstddef>
#include <cstdint>
#include <vector>

struct llama_model;
struct llama_hparams;

void dflash_validate_config(
        uint32_t block_size,
        uint32_t mask_token_id,
        uint32_t vocab_size,
        uint32_t target_layer_count,
        const std::vector<int> & target_layers);

void dflash_append_target_features(
        std::vector<float> & destination,
        const std::vector<const float *> & layers,
        size_t n_tokens,
        size_t n_embd);

void dflash_configure_swa(
        llama_hparams & hparams,
        uint32_t sliding_window,
        const std::vector<bool> & pattern);

void dflash_expand_logits(
        std::vector<float> & target_logits,
        const std::vector<float> & draft_logits,
        const std::vector<int32_t> & draft_to_target_delta,
        size_t target_vocab_size);

struct llm_build_dflash_encode : public llm_graph_context {
    llm_build_dflash_encode(const llama_model & model, const llm_graph_params & params);
};

struct llm_build_dflash_decode : public llm_graph_context {
    llm_build_dflash_decode(const llama_model & model, const llm_graph_params & params);
};
