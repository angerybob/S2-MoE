#pragma once

#include "llama.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()         const;
    uint32_t n_ctx_per_seq() const;
    uint32_t n_batch()       const;
    uint32_t n_ubatch()      const;
    uint32_t n_seq_max()     const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    // EAGLE specific methods for hidden states management
    void set_target_hidden_states(const float * hidden_states, size_t size);
    const float * get_target_hidden_states() const;
    size_t get_target_hidden_states_size() const;
    bool has_target_hidden_states() const;

    void clear_target_hidden_states();

    // EAGLE3: Draft model's own hidden states methods
    void set_draft_hidden_states(const float * hidden_states, size_t size);
    const float * get_draft_hidden_states() const;
    size_t get_draft_hidden_states_size() const;
    bool has_draft_hidden_states() const;
    void clear_draft_hidden_states();

    void set_eagle3_g_embeddings(const float * g_embd, int32_t n_embd, int32_t n_tokens);
    const float * get_eagle3_g_embeddings() const;
    size_t get_eagle3_g_embeddings_size() const;
    const float * get_eagle3_target_features() const;
    size_t get_eagle3_target_features_size() const;
    void select_eagle3_target_features(const int32_t * indices, int32_t n_indices);

    // EAGLE3: Auto-update mechanism for draft hidden states
    void auto_update_draft_hidden_states();

    void set_draft_context(bool is_draft) {
        is_draft_context = is_draft;
        cparams.is_draft_context = is_draft;
        graph_reuse_disable = true;
    }
    bool is_draft_ctx() const { return is_draft_context; }

    void set_moe_reuse_verify(bool enabled, bool runtime_strength) {
        moe_reuse_verify_enabled = enabled;
        moe_reuse_runtime_strength = runtime_strength;
        graph_reuse_disable = true;
    }

    
    // EAGLE3: Store last computed hidden states for draft model access
    void set_last_hidden_states(const float * hidden_states, size_t hidden_size, int32_t n_tokens);
    const float * get_last_hidden_states() const;
    size_t get_last_hidden_states_size() const;
    int32_t get_last_hidden_states_tokens() const;
    bool has_last_hidden_states() const;

    // EAGLE3: Get current graph result for hidden state extraction
    llm_graph_result * get_graph_result() { return gf_res_prev.get(); }

    void set_moe_topk_enabled(bool enabled);
    bool get_last_moe_topk(int32_t token_index, llama_moe_topk_layer * out_layers, int32_t max_layers, int32_t * out_n_layers) const;

    void set_adapter_lora(
            llama_adapter_lora * adapter,
            float scale);

    bool rm_adapter_lora(
            llama_adapter_lora * adapter);

    void clear_adapter_lora();

    // EAGLE3: Share token embedding tensor from target model
    struct ggml_tensor * shared_token_embd = nullptr;

    // EAGLE3: GPU-compatible embedding tensor storage for dual-model loading
    std::vector<float> embedding_weights;  // Complete embedding matrix for CPU lookup
    std::unique_ptr<float[]> current_embedding_buffer;  // Current token embedding buffer
    size_t current_embedding_size = 0;  // Size of current embedding buffer
    size_t embedding_dim = 0;  // Embedding dimension
    size_t vocab_size = 0;  // Vocabulary size
    struct ggml_tensor * draft_embedding_tensor = nullptr;  // Pseudo embedding tensor for graph

    // EAGLE3: Dual-model loading helper methods
    void create_draft_embedding_tensor(const struct ggml_tensor* target_embd);

    bool apply_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    // EAGLE3 SMP: Extract and accumulate hidden states for current batch
    void extract_and_accumulate_hidden_states(const struct llm_graph_result * res);

    // EAGLE3 SMP: Current batch EAGLE hidden states storage
    // Key: layer_idx, Value: vector of current batch tokens' HS for that layer
    std::unordered_map<int, std::vector<float>> current_batch_eagle_hs;

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    //
    // graph
    //

public:
    uint32_t graph_max_nodes() const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);
    void extract_eagle3_features(const llama_ubatch & ubatch);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false);
    // 修改为:
    llama_memory_i * memory = nullptr; 

    bool owns_memory = true; 
private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams       cparams;
    llama_adapter_cvec  cvec;
    llama_adapter_loras loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    // 原本是: std::unique_ptr<llama_memory_i> memory;
    // 修改为:
//     llama_memory_i * memory = nullptr; 

//     bool owns_memory = true; 
// 迁移到public部分

    // EAGLE draft context flag
    bool is_draft_context = false;

    // EAGLE hidden states for target model (stored for draft model usage)
    std::vector<float> target_hidden_states;
    size_t hidden_states_size = 0;  // n_embd * n_tokens
    bool has_hidden_states = false;

    // EAGLE3: Draft model's own hidden states (2560 dims from final layer)
    std::vector<float> draft_hidden_states;    // 2560维，存储DFT最后一层输出
    size_t draft_hidden_states_size = 0;       // 固定2560
    bool has_draft_hidden_states_data = false;

    std::vector<float> eagle3_g_embeddings;
    mutable std::vector<int> eagle3_extract_layer_indices;
    mutable std::vector<ggml_tensor *> eagle3_extract_tensors;
    std::vector<float> eagle3_target_features;
    bool moe_reuse_verify_enabled = false;
    bool moe_reuse_runtime_strength = false;

    // EAGLE3: Last computed hidden states for draft model access
    std::vector<float> last_hidden_states;
    size_t last_hidden_size = 0;     // n_embd
    int32_t last_n_tokens = 0;       // number of tokens
    bool has_last_hidden = false;

    
    
    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    struct moe_topk_layer_data {
        int layer = -1;
        int n_expert = 0;
        std::vector<int32_t> expert_ids;
    };

    void clear_moe_topk(uint32_t n_tokens);
    void extract_moe_topk_from_ubatch(const llm_graph_result * res, uint32_t n_tokens_all);

    bool moe_topk_enabled = false;
    bool moe_topk_initialized = false;
    uint32_t moe_topk_last_n_tokens = 0;
    std::vector<moe_topk_layer_data> moe_topk_layers;

    // buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
