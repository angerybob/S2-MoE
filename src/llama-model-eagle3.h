#pragma once

#include "llama-graph.h"

struct llama_model;

struct llm_build_eagle3_encode : public llm_graph_context {
    llm_build_eagle3_encode(const llama_model & model, const llm_graph_params & params);

private:
    ggml_tensor * build_inp_embd() const;
};

struct llm_build_eagle3_decode : public llm_graph_context {
    llm_build_eagle3_decode(const llama_model & model, const llm_graph_params & params);
};
