#pragma once

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace next_spec {

inline void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void disable_graphs() {
    // Capture changes the graph's state-copy destinations. Graph reuse and CUDA
    // graph capture have not been validated for this experimental entry point.
#ifdef _WIN32
    _putenv_s("GGML_CUDA_DISABLE_GRAPHS", "1");
    _putenv_s("LLAMA_GRAPH_REUSE_DISABLE", "1");
#else
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
    setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
#endif
}

inline void enable_q8_cuda_ar_compatibility() {
    // Quantized CUDA batch projections otherwise use a different activation
    // arithmetic from serial AR. Keep this policy local to the hybrid tools.
#ifdef _WIN32
    _putenv_s("GGML_CUDA_Q8_AR_COMPAT", "1");
#else
    setenv("GGML_CUDA_Q8_AR_COMPAT", "1", 1);
#endif
}

inline void decode(llama_context * ctx, const std::vector<llama_token> & tokens,
                   llama_pos first, bool all_logits = false) {
    require(!tokens.empty(), "cannot decode an empty batch");
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    batch.n_tokens = tokens.size();
    for (int i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = first + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = all_logits || i + 1 == batch.n_tokens;
    }
    const int result = llama_decode(ctx, batch);
    llama_batch_free(batch);
    require(result == 0, "Qwen3-Next decode failed");
}

struct snapshot {
    llama_pos first;
    std::vector<uint8_t> data;

    explicit snapshot(llama_context * ctx) {
        llama_synchronize(ctx);
        first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
        data.resize(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
        require(!data.empty(), "recurrent snapshot is unavailable");
        require(llama_state_seq_get_data_ext(ctx, data.data(), data.size(), 0,
                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == data.size(), "snapshot capture failed");
    }

    void restore(llama_context * ctx) const {
        llama_synchronize(ctx);
        require(llama_state_seq_set_data_ext(ctx, data.data(), data.size(), 0,
                    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == data.size(), "snapshot restore failed");
        require(llama_memory_seq_rm(llama_get_memory(ctx), 0, first, -1), "KV tail removal failed");
    }
};

inline llama_token greedy(llama_context * ctx, int index = -1) {
    const auto n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * logits = llama_get_logits_ith(ctx, index);
    require(logits != nullptr, "missing verification logits");
    for (int i = 0; i < n_vocab; ++i) {
        require(std::isfinite(logits[i]), "non-finite logits");
    }
    return std::max_element(logits, logits + n_vocab) - logits;
}

} // namespace next_spec
