#include "llama-impl.h"

#include "llama-chat.h"
#include "llama-mmap.h"
#include "llama-vocab.h"
#include "llama-model-loader.h"
#include "llama-model-saver.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include <vector>
#include <limits>
#include <memory>
#include <thread>
#include <cstdlib>
#include <string>
#include <unordered_map>
// --- INSERT START: Global SSD Bridge ---
#include <map>
#include <mutex>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif
#ifdef GGML_USE_CUDA
extern "C" void ggml_cuda_register_ssd_expert(const char* key, const char* fname, size_t offset, size_t size, void* data_ptr);
extern "C" int llama_cuda_domino_sample(
        const struct ggml_tensor * target_tok_embd,
        const struct ggml_tensor * target_output,
        const struct ggml_tensor * gru_w_ih,
        const struct ggml_tensor * gru_w_hh,
        const struct ggml_tensor * fc1,
        const struct ggml_tensor * fc2,
        const struct ggml_tensor * parallel_hidden,
        int32_t output_index,
        const int32_t * prefix_tokens,
        int32_t n_prefix_tokens,
        bool apply_correction,
        int32_t * out_token);
#else
extern "C" void ggml_cuda_register_ssd_expert(const char*, const char*, size_t, size_t, void*) {}
#endif
//
// interface implementation
//

// [新增] 用于实现共享 KV Cache 的接口
void llama_share_kv_cache(struct llama_context * ctx_draft, struct llama_context * ctx_target) {
    if (!ctx_draft || !ctx_target) return;
    
    // 1. 如果 Draft 原本有内存（自己申请的），先释放掉
    if (ctx_draft->owns_memory && ctx_draft->memory) {
        delete ctx_draft->memory;
    }

    // 2. 指针“寄生”：让 Draft 直接指向 Target 的内存对象
    ctx_draft->memory = ctx_target->memory;

    // 3. 标记 Draft 不拥有内存（防止析构时 double free）
    ctx_draft->owns_memory = false;

    // 打印日志确认
    LLAMA_LOG_INFO("%s: Success. Draft context is now sharing KV cache with Target.\n", __func__);
}

const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:
            return "auto";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED:
            return "disabled";
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:
            return "enabled";
    }
    GGML_ABORT("fatal error");
}

struct llama_sampler_chain_params llama_sampler_chain_default_params() {
    struct llama_sampler_chain_params result = {
        /*.no_perf                     =*/ true,
    };

    return result;
}

size_t llama_max_devices(void) {
    return 16;
}

bool llama_supports_mmap(void) {
    return llama_mmap::SUPPORTED;
}

bool llama_supports_mlock(void) {
    return llama_mlock::SUPPORTED;
}

bool llama_supports_gpu_offload(void) {
    return ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) != nullptr ||
           ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU) != nullptr ||
           llama_supports_rpc();
}

bool llama_supports_rpc(void) {
    return ggml_backend_reg_by_name("RPC") != nullptr;
}

void llama_backend_init(void) {
    ggml_time_init();

    // needed to initialize f16 tables
    {
        struct ggml_init_params params = { 0, NULL, false };
        struct ggml_context * ctx = ggml_init(params);
        ggml_free(ctx);
    }
}

void llama_numa_init(enum ggml_numa_strategy numa) {
    if (numa != GGML_NUMA_STRATEGY_DISABLED) {
        auto * dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        GGML_ASSERT(dev && "CPU backend is not loaded");
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * numa_init_fn = (decltype(ggml_numa_init) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_numa_init");
        if (numa_init_fn) {
            numa_init_fn(numa);
        }
    }
}

void llama_backend_free(void) {
    ggml_quantize_free();
}

int64_t llama_time_us(void) {
    return ggml_time_us();
}

// Returns 0 on success, -1 on error, and -2 on cancellation via llama_progress_callback
static int llama_model_load(const std::string & fname, std::vector<std::string> & splits, llama_model & model, llama_model_params & params) {
    // loading time will be recalculated after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = 0;
    time_meas tm(model.t_load_us);

    model.t_start_us = tm.t_start_us;

    try {
        llama_model_loader ml(fname, splits, params.use_mmap, params.check_tensors, params.use_ssd_moe ,params.kv_overrides, params.tensor_buft_overrides,params.hot_experts_path);

        ml.print_info();

        model.hparams.vocab_only = params.vocab_only;

        try {
            model.load_arch(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model architecture: " + std::string(e.what()));
        }
        try {
            model.load_hparams(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model hyperparameters: " + std::string(e.what()));
        }
        try {
            model.load_vocab(ml);
        } catch(const std::exception & e) {
            throw std::runtime_error("error loading model vocabulary: " + std::string(e.what()));
        }

        model.load_stats(ml);
        model.print_info();

        if (params.vocab_only) {
            LLAMA_LOG_INFO("%s: vocab only - skipping tensors\n", __func__);
            return 0;
        }

        if (!model.load_tensors(ml)) {
            return -2;
        }
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading model: %s\n", __func__, err.what());
        return -1;
    }

    return 0;
}

static struct llama_model * llama_model_load_from_file_impl(
        const std::string & path_model,
        std::vector<std::string> & splits,
        struct llama_model_params params) {
    ggml_time_init();

    if (!params.vocab_only && ggml_backend_reg_count() == 0) {
        LLAMA_LOG_ERROR("%s: no backends are loaded. hint: use ggml_backend_load() or ggml_backend_load_all() to load a backend before calling this function\n", __func__);
        return nullptr;
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                LLAMA_LOG_CONT(".");
                if (percentage >= 100) {
                    LLAMA_LOG_CONT("\n");
                }
            }
            return true;
        };
    }

    llama_model * model = new llama_model(params);

    // create list of devices to use with this model
    if (params.devices) {
        for (ggml_backend_dev_t * dev = params.devices; *dev; ++dev) {
            model->devices.push_back(*dev);
        }
    } else {
        // default device selection

        // build list of available devices
        std::vector<ggml_backend_dev_t> gpus;
        std::vector<ggml_backend_dev_t> igpus;
        std::vector<ggml_backend_dev_t> rpc_servers;

        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            switch (ggml_backend_dev_type(dev)) {
                case GGML_BACKEND_DEVICE_TYPE_CPU:
                case GGML_BACKEND_DEVICE_TYPE_ACCEL:
                    // skip CPU backends since they are handled separately
                    break;

                case GGML_BACKEND_DEVICE_TYPE_GPU: {
                    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
                    if (ggml_backend_reg_name(reg) == std::string("RPC")) {
                        rpc_servers.push_back(dev);
                    } else {
                        // check if there is already a GPU with the same device id
                        ggml_backend_dev_props props;
                        ggml_backend_dev_get_props(dev, &props);
                        auto it = std::find_if(gpus.begin(), gpus.end(), [&props](ggml_backend_dev_t d) {
                            ggml_backend_dev_props d_props;
                            ggml_backend_dev_get_props(d, &d_props);
                            if (props.device_id && d_props.device_id) {
                                return strcmp(props.device_id, d_props.device_id) == 0;
                            }
                            return false;
                        });

                        if (it != gpus.end()) {
                            LLAMA_LOG_INFO("%s: skipping device %s (%s) with id %s - already using device %s (%s) with the same id\n",
                                    __func__,
                                    ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                                    props.device_id ? props.device_id : "unknown id",
                                    ggml_backend_dev_name(*it), ggml_backend_dev_description(*it));
                        } else {
                            gpus.push_back(dev);
                        }
                    }
                    break;
                }

                case GGML_BACKEND_DEVICE_TYPE_IGPU:
                    igpus.push_back(dev);
                    break;
            }
        }

        // add RPC servers at the front of the list to minimize network transfers
        model->devices.insert(model->devices.begin(), rpc_servers.begin(), rpc_servers.end());

        // add GPUs
        model->devices.insert(model->devices.end(), gpus.begin(), gpus.end());

        // add integrated GPUs only if no other devices were found
        if (model->devices.empty()) {
            model->devices.insert(model->devices.end(), igpus.begin(), igpus.end());
        }
    }

    // if using single GPU mode, remove all except the main GPU
    if (params.split_mode == LLAMA_SPLIT_MODE_NONE) {
        if (params.main_gpu < 0) {
            model->devices.clear();
        } else {
            if (params.main_gpu >= (int)model->devices.size()) {
                LLAMA_LOG_ERROR("%s: invalid value for main_gpu: %d (available devices: %zu)\n", __func__, params.main_gpu, model->devices.size());
                llama_model_free(model);
                return nullptr;
            }
            ggml_backend_dev_t main_gpu = model->devices[params.main_gpu];
            model->devices.clear();
            model->devices.push_back(main_gpu);
        }
    }

    for (auto * dev : model->devices) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        LLAMA_LOG_INFO("%s: using device %s (%s) (%s) - %zu MiB free\n", __func__,
                ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                props.device_id ? props.device_id : "unknown id",
                props.memory_free/1024/1024);
    }

    const int status = llama_model_load(path_model, splits, *model, params);
// --- INSERT START: Sync to Global ---
    if (status >= 0) {
        auto experts = model->get_ssd_experts();
        int count_hot = 0;
        int count_ssd = 0;

        for (auto const& [key, val] : experts) {
            std::string fname;
            if (val.file_idx < splits.size()) {
                fname = splits[val.file_idx];
            } else {
                fname = path_model;
            }
            
            // 统计数量
            if (val.data_ptr != nullptr) {
                count_hot++;
            } else {
                count_ssd++;
            }

            ggml_cuda_register_ssd_expert(key.c_str(), fname.c_str(), val.file_offs, val.size, val.data_ptr);
        }
        
        // 修改日志输出，把混合状态打印出来
        LLAMA_LOG_INFO("%s: Registered %zu experts total (%d Hot/RAM, %d Cold/SSD) to CUDA backend\n", 
                       __func__, experts.size(), count_hot, count_ssd);
    }
    // --- INSERT END ---


    GGML_ASSERT(status <= 0);
    if (status < 0) {
        if (status == -1) {
            LLAMA_LOG_ERROR("%s: failed to load model\n", __func__);
        } else if (status == -2) {
            LLAMA_LOG_INFO("%s: cancelled model load\n", __func__);
        }

        llama_model_free(model);
        return nullptr;
    }

    return model;
}

// deprecated
struct llama_model * llama_load_model_from_file(
        const char * path_model,
        struct llama_model_params params) {
    return llama_model_load_from_file(path_model, params);
}

struct llama_model * llama_model_load_from_file(
        const char * path_model,
        struct llama_model_params params) {
    std::vector<std::string> splits = {};
    return llama_model_load_from_file_impl(path_model, splits, params);
}

struct llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        struct llama_model_params params) {
    std::vector<std::string> splits;
    if (n_paths == 0) {
        LLAMA_LOG_ERROR("%s: list of splits is empty\n", __func__);
        return nullptr;
    }
    for (size_t i = 0; i < n_paths; ++i) {
        splits.push_back(paths[i]);
    }
    return llama_model_load_from_file_impl(splits.front(), splits, params);
}

void llama_model_save_to_file(const struct llama_model * model, const char * path_model) {
    llama_model_saver ms(*model);
    ms.add_kv_from_model();
    ms.add_tensors_from_model();
    ms.save(path_model);
}

//
// chat templates
//

int32_t llama_chat_apply_template(
                              const char * tmpl,
         const struct llama_chat_message * chat,
                                  size_t   n_msg,
                                    bool   add_ass,
                                    char * buf,
                                 int32_t   length) {
    const std::string curr_tmpl(tmpl == nullptr ? "chatml" : tmpl);

    // format the chat to string
    std::vector<const llama_chat_message *> chat_vec;
    chat_vec.resize(n_msg);
    for (size_t i = 0; i < n_msg; i++) {
        chat_vec[i] = &chat[i];
    }

    std::string formatted_chat;
    llm_chat_template detected_tmpl = llm_chat_detect_template(curr_tmpl);
    if (detected_tmpl == LLM_CHAT_TEMPLATE_UNKNOWN) {
        return -1;
    }
    int32_t res = llm_chat_apply_template(detected_tmpl, chat_vec, formatted_chat, add_ass);
    if (res < 0) {
        return res;
    }
    if (buf && length > 0) {
        strncpy(buf, formatted_chat.c_str(), length);
    }
    return res;
}

//
// model split
//

int llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int split_no, int split_count) {
    static const char * const SPLIT_PATH_FORMAT = "%s-%05d-of-%05d.gguf";
    if (snprintf(split_path, maxlen, SPLIT_PATH_FORMAT, path_prefix, split_no + 1, split_count)) {
        return strlen(split_path);
    }
    return 0;
}

int llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int split_no, int split_count) {
    std::string str_split_path(split_path);
    char postfix[32];
    snprintf(postfix, 32, "-%05d-of-%05d.gguf", split_no + 1, split_count);
    std::string str_postfix(postfix);

    // check if split_prefix ends with postfix
    int size_prefix = str_split_path.size() - str_postfix.size();
    if (size_prefix > 0 && str_split_path.find(str_postfix, size_prefix) != std::string::npos) {
        snprintf(split_prefix, std::min((size_t) size_prefix + 1, maxlen), "%s", split_path);
        return size_prefix;
    }

    return 0;
}

const char * llama_print_system_info(void) {
    static std::string s;
    s.clear(); // Clear the string, since it's static, otherwise it will accumulate data from previous calls.

    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto * reg = ggml_backend_reg_get(i);
        auto * get_features_fn = (ggml_backend_get_features_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_get_features");
        if (get_features_fn) {
            ggml_backend_feature * features = get_features_fn(reg);
            s += ggml_backend_reg_name(reg);
            s += " : ";
            for (; features->name; features++) {
                s += features->name;
                s += " = ";
                s += features->value;
                s += " | ";
            }
        }
    }

    return s.c_str();
}

//
// EAGLE3: Hidden states extraction - C API implementations
//

int32_t llama_context_get_hidden_states_layer_count(struct llama_context * ctx) {
    if (!ctx) {
        return -1;
    }

    // Get the current graph result
    struct llm_graph_result * result = ctx->get_graph_result();
    if (!result) {
        return 0;
    }

    return (int32_t)result->get_captured_layer_count();
}

const int * llama_context_get_hidden_states_layer_indices(struct llama_context * ctx, int32_t * out_count) {
    if (!ctx || !out_count) {
        if (out_count) *out_count = 0;
        return nullptr;
    }

    // Get the current graph result
    struct llm_graph_result * result = ctx->get_graph_result();
    if (!result) {
        *out_count = 0;
        return nullptr;
    }

    *out_count = (int32_t)result->captured_indices.size();
    return result->captured_indices.data();
}

float * llama_context_extract_hidden_states_from_layer(struct llama_context * ctx, int32_t token_index, int32_t layer_idx, int32_t * out_n_embd) {
    if (!ctx || !out_n_embd) {
        return nullptr;
    }

    // Get the current graph result
    struct llm_graph_result * result = ctx->get_graph_result();
    if (!result) {
        return nullptr;
    }

    // Get the map of captured layers
    auto layer_map = result->get_captured_layers_map();

    // Find the tensor for this layer
    auto it = layer_map.find(layer_idx);
    if (it == layer_map.end()) {
        // Layer not captured
        *out_n_embd = 0;
        return nullptr;
    }

    ggml_tensor * hidden_states = it->second;
    if (!hidden_states) {
        *out_n_embd = 0;
        return nullptr;
    }

    // Get dimensions
    const int64_t n_embd = hidden_states->ne[0];
    const int64_t n_tokens = hidden_states->ne[1];

    // Check if token_index is valid
    if (token_index < 0 || token_index >= n_tokens) {
        *out_n_embd = 0;
        return nullptr;
    }

    // Allocate output buffer
    float * output = (float *)malloc(n_embd * sizeof(float));
    if (!output) {
        *out_n_embd = 0;
        return nullptr;
    }

    // Extract the hidden state for the specified token
    // hidden_states is [n_embd, n_tokens], we want column token_index
    const size_t row_size = ggml_row_size(hidden_states->type, n_embd);
    const size_t offset = token_index * row_size;

    // Copy the data
    ggml_backend_tensor_get(hidden_states, output, offset, row_size);

    *out_n_embd = (int32_t)n_embd;
    return output;
}

// EAGLE3 SMP: Extract EAGLE hidden states for a specific token in the current batch
float * llama_context_get_eagle_hs_at_batch_idx(struct llama_context * ctx, int batch_idx, int32_t * out_dim) {
    if (!ctx || batch_idx < 0 || !out_dim) {
        return nullptr;
    }

    // Get model parameters for dynamic calculation
    const struct llama_model * model = llama_get_model(ctx);
    const int32_t n_embd = model->hparams.n_embd;
    const int n_layer = model->hparams.n_layer;

    // Dynamic EAGLE layers calculation: {2, n_layer/2, n_layer-3}
    const int eagle_layers[] = {2, n_layer/2, n_layer-3};
    const int num_layers = 3;

    // **关键修复**：如果有最新的GPU结果，强制进行GPU->CPU同步
    auto * graph_result = ctx->get_graph_result();
    if (graph_result) {
        // 清除过时的CPU缓存
        for (int layer_idx : eagle_layers) {
            ctx->current_batch_eagle_hs[layer_idx].clear();
        }

        // 从GPU重新提取数据，确保数据新鲜
        ctx->extract_and_accumulate_hidden_states(graph_result);

        LLAMA_LOG_DEBUG("EAGLE3: 为batch_idx %d强制GPU->CPU同步\n", batch_idx);
    }

    // Allocate output buffer: [layers * n_embd]
    // This contains all EAGLE layer data for one token
    float * output = (float *)malloc(num_layers * n_embd * sizeof(float));
    if (!output) {
        return nullptr;
    }

    for (int i = 0; i < num_layers; i++) {
        int layer_id = eagle_layers[i];
        auto it = ctx->current_batch_eagle_hs.find(layer_id);

        if (it == ctx->current_batch_eagle_hs.end()) {
            free(output);
            return nullptr; // Layer not found
        }

        // Calculate offset: batch_idx * n_embd
        // Check bounds (use size_t to avoid integer overflow)
        if ((size_t)(batch_idx + 1) * n_embd > it->second.size()) {
            free(output);
            return nullptr; // batch_idx out of bounds
        }

        // Copy data
        const float* src = it->second.data() + (batch_idx * n_embd);
        float* dst = output + (i * n_embd);
        memcpy(dst, src, n_embd * sizeof(float));
    }

    *out_dim = num_layers * n_embd;
    return output;
}

struct llm_graph_result * llama_context_get_graph_result(struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }
    return ctx->get_graph_result();
}

void llama_context_set_target_hidden_states(struct llama_context * ctx, const float * hidden_states, size_t size) {
    LLAMA_LOG_INFO("EAGLE3: llama_context_set_target_hidden_states called - ctx=%p, hidden_states=%p, size=%zu\n",
                   (void*)ctx, (const void*)hidden_states, size);

    if (ctx && hidden_states) {
        // Log complete range for debugging - check all elements
        float hs_min = hidden_states[0], hs_max = hidden_states[0];
        for (size_t i = 1; i < size; i++) {
            hs_min = std::min(hs_min, hidden_states[i]);
            hs_max = std::max(hs_max, hidden_states[i]);
        }
        LLAMA_LOG_INFO("EAGLE3: hidden_states input stats - min: %.6f, max: %.6f (full range, %zu elements)\n", hs_min, hs_max, size);

        ctx->set_target_hidden_states(hidden_states, size);
        LLAMA_LOG_INFO("EAGLE3: set_target_hidden_states completed successfully\n");
    } else {
        LLAMA_LOG_INFO("EAGLE3: set_target_hidden_states failed - ctx=%p, hidden_states=%p\n",
                       (void*)ctx, (const void*)hidden_states);
    }
}

const float * llama_context_get_target_hidden_states(const struct llama_context * ctx) {
    return ctx ? ctx->get_target_hidden_states() : nullptr;
}

size_t llama_context_get_target_hidden_states_size(const struct llama_context * ctx) {
    return ctx ? ctx->get_target_hidden_states_size() : 0;
}

bool llama_context_has_target_hidden_states(const struct llama_context * ctx) {
    return ctx ? ctx->has_target_hidden_states() : false;
}

void llama_context_clear_target_hidden_states(struct llama_context * ctx) {
    if (ctx) {
        ctx->clear_target_hidden_states();
    }
}

//
// EAGLE3: Draft model hidden states public API
//

void llama_context_set_draft_hidden_states(struct llama_context * ctx, const float * hidden_states, size_t size) {
    LLAMA_LOG_INFO("EAGLE3: llama_context_set_draft_hidden_states called - ctx=%p, hidden_states=%p, size=%zu\n",
                   (void*)ctx, (const void*)hidden_states, size);

    if (ctx && hidden_states) {
        // Log some sample values for debugging
        float hs_min = hidden_states[0], hs_max = hidden_states[0];
        for (int i = 1; i < std::min(int(size), 100); i++) {
            hs_min = std::min(hs_min, hidden_states[i]);
            hs_max = std::max(hs_max, hidden_states[i]);
        }
        LLAMA_LOG_INFO("EAGLE3: draft hidden_states input stats - min: %.6f, max: %.6f\n", hs_min, hs_max);

        ctx->set_draft_hidden_states(hidden_states, size);
        LLAMA_LOG_INFO("EAGLE3: set_draft_hidden_states completed successfully\n");
    } else {
        LLAMA_LOG_INFO("EAGLE3: set_draft_hidden_states failed - ctx=%p, hidden_states=%p\n",
                       (void*)ctx, (const void*)hidden_states);
    }
}

const float * llama_context_get_draft_hidden_states(const struct llama_context * ctx) {
    return ctx ? ctx->get_draft_hidden_states() : nullptr;
}

size_t llama_context_get_draft_hidden_states_size(const struct llama_context * ctx) {
    return ctx ? ctx->get_draft_hidden_states_size() : 0;
}

bool llama_context_has_draft_hidden_states(const struct llama_context * ctx) {
    return ctx ? ctx->has_draft_hidden_states() : false;
}

void llama_context_clear_draft_hidden_states(struct llama_context * ctx) {
    if (ctx) {
        ctx->clear_draft_hidden_states();
    }
}

void llama_context_set_draft_context(struct llama_context * ctx, bool is_draft) {
    if (ctx) {
        ctx->set_draft_context(is_draft);
        LLAMA_LOG_INFO("EAGLE3: Set context %p as draft context: %s\n",
                       (void*)ctx, is_draft ? "true" : "false");
    }
}

bool llama_context_is_draft_context(const struct llama_context * ctx) {
    return ctx ? ctx->is_draft_ctx() : false;
}

void llama_context_set_draft_skip_layers(
        struct llama_context * ctx,
        const int32_t * attn_layers, size_t attn_count,
        const int32_t * mlp_layers,  size_t mlp_count) {
    if (!ctx) {
        return;
    }
    std::vector<int32_t> attn;
    std::vector<int32_t> mlp;
    if (attn_layers && attn_count > 0) {
        attn.assign(attn_layers, attn_layers + attn_count);
    }
    if (mlp_layers && mlp_count > 0) {
        mlp.assign(mlp_layers, mlp_layers + mlp_count);
    }
    ctx->set_draft_skip_layers(attn, mlp);
}

void llama_context_set_draft_expert_topk(struct llama_context * ctx, int32_t topk) {
    if (ctx) {
        ctx->set_draft_expert_topk(topk);
    }
}

void llama_context_set_moe_topk_log_k(struct llama_context * ctx, int32_t topk) {
    if (ctx) {
        ctx->set_moe_topk_log_k(topk);
    }
}

void llama_context_set_draft_layer_expert_topk(
        struct llama_context * ctx,
        const int32_t * layer_topk, size_t layer_count) {
    if (!ctx) {
        return;
    }
    std::vector<int32_t> layer;
    if (layer_topk && layer_count > 0) {
        layer.assign(layer_topk, layer_topk + layer_count);
    }
    ctx->set_draft_layer_expert_topk(layer);
}

void llama_context_clear_draft_layer_expert_topk(struct llama_context * ctx) {
    if (ctx) {
        ctx->clear_draft_layer_expert_topk();
    }
}

void llama_context_set_moe_expert_topk(struct llama_context * ctx, int32_t topk) {
    if (ctx) {
        ctx->set_moe_expert_topk(topk);
    }
}

void llama_context_set_moe_reuse_verify(
        struct llama_context * ctx,
        bool enabled,
        bool runtime_strength) {
    if (ctx) {
        ctx->set_moe_reuse_verify(enabled, runtime_strength);
    }
}

void llama_set_moe_topk(struct llama_context * ctx, bool enabled) {
    if (ctx) {
        ctx->set_moe_topk_enabled(enabled);
    }
}

bool llama_get_last_moe_topk(
        struct llama_context      * ctx,
        int32_t                     token_index_in_last_decode_batch,
        struct llama_moe_topk_layer * out_layers,
        int32_t                     max_layers,
        int32_t                   * out_n_layers) {
    return ctx ? ctx->get_last_moe_topk(token_index_in_last_decode_batch, out_layers, max_layers, out_n_layers) : false;
}

void llama_context_set_target_embedding_layer(struct llama_context * ctx_draft, const struct llama_context * ctx_tgt) {
    LLAMA_LOG_INFO("EAGLE3: llama_context_set_target_embedding_layer called - draft_ctx=%p, target_ctx=%p\n",
                   (void*)ctx_draft, (const void*)ctx_tgt);

    if (!ctx_draft || !ctx_tgt) {
        LLAMA_LOG_ERROR("EAGLE3: set_target_embedding_layer failed - invalid parameters\n");
        return;
    }

    const struct llama_model * model_draft = llama_get_model(ctx_draft);
    const struct llama_model * model_tgt = llama_get_model(ctx_tgt);

    if (model_draft->arch != LLM_ARCH_EAGLE) {
        LLAMA_LOG_ERROR("EAGLE3: set_target_embedding_layer failed - draft model is not EAGLE architecture\n");
        return;
    }

    // Get token embedding tensor from target model
    struct ggml_tensor * target_embd_tensor = model_tgt->tok_embd;

    if (!target_embd_tensor) {
        LLAMA_LOG_ERROR("EAGLE3: Failed to find token embedding tensor in target model\n");
        return;
    }

    // Set the shared token embedding tensor in draft context
    ctx_draft->shared_token_embd = target_embd_tensor;

    LLAMA_LOG_INFO("EAGLE3: Successfully set shared token embedding tensor - shape: [%lld, %lld]\n",
                   (long long)target_embd_tensor->ne[0], (long long)target_embd_tensor->ne[1]);
}

struct ggml_tensor * llama_context_get_shared_token_embd(const struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }
    return ctx->shared_token_embd;
}

void llama_context_clear_shared_token_embd(struct llama_context * ctx) {
    if (!ctx) {
        return;
    }
    ctx->shared_token_embd = nullptr;
    LLAMA_LOG_INFO("EAGLE3: Cleared shared token embedding tensor\n");
}


void llama_context_compute_token_embedding_cpu(struct llama_context * ctx_dft, struct llama_context * ctx_tgt_cpu, llama_token token_id) {
    if (!ctx_dft || !ctx_tgt_cpu) {
        LLAMA_LOG_ERROR("EAGLE3: Invalid contexts for token embedding computation\n");
        return;
    }

    // Get CPU target model and its embedding tensor
    const llama_model * model_tgt = llama_get_model(ctx_tgt_cpu);
    if (!model_tgt) {
        LLAMA_LOG_ERROR("EAGLE3: Failed to get CPU target model\n");
        return;
    }

    struct ggml_tensor * target_embd_tensor = model_tgt->tok_embd;
    if (!target_embd_tensor || !target_embd_tensor->data) {
        LLAMA_LOG_ERROR("EAGLE3: Failed to access CPU target embedding tensor\n");
        return;
    }

    // Get dimensions
    size_t n_embd = model_tgt->hparams.n_embd;
    const struct llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    size_t n_vocab = llama_vocab_n_tokens(vocab);

    if (token_id >= (int)n_vocab) {
        LLAMA_LOG_ERROR("EAGLE3: Token ID %d out of range for CPU target model vocab size %zu\n", token_id, n_vocab);
        return;
    }

    // Direct lookup from CPU embedding matrix
    const float* embedding_matrix = (const float*)target_embd_tensor->data;
    const float * src_embedding = embedding_matrix + (size_t)token_id * n_embd;

    // Store in draft context's current embedding buffer for graph computation
    if (!ctx_dft->current_embedding_buffer || n_embd != ctx_dft->current_embedding_size) {
        ctx_dft->current_embedding_buffer.reset(new float[n_embd]);
        ctx_dft->current_embedding_size = n_embd;
    }

    std::copy(src_embedding, src_embedding + n_embd, ctx_dft->current_embedding_buffer.get());

    printf("[DEBUG] CPU computed embedding for token %d: [%.3f, %.3f, %.3f, ...] (direct CPU lookup)\n",
           token_id,
           src_embedding[0],
           src_embedding[1],
           src_embedding[2]);
}

// EAGLE3: Vocabulary mapping APIs
const int32_t * llama_model_get_eagle_d2t_map(const struct llama_model * model) {
    if (!model || model->arch != LLM_ARCH_EAGLE) {
        return nullptr;
    }
    return model->ea_layer.d2t_map.data();
}

const uint8_t * llama_model_get_eagle_t2d_map(const struct llama_model * model) {
    if (!model || model->arch != LLM_ARCH_EAGLE) {
        return nullptr;
    }
    return model->ea_layer.t2d_map.data();
}

size_t llama_model_get_eagle_d2t_size(const struct llama_model * model) {
    if (!model || model->arch != LLM_ARCH_EAGLE) {
        return 0;
    }
    return model->ea_layer.d2t_map.size();
}

size_t llama_model_get_eagle_t2d_size(const struct llama_model * model) {
    if (!model || model->arch != LLM_ARCH_EAGLE) {
        return 0;
    }
    return model->ea_layer.t2d_map.size();
}

bool llama_model_has_eagle_vocab_mapping(const struct llama_model * model) {
    return (model && model->arch == LLM_ARCH_EAGLE &&
            !model->ea_layer.d2t_map.empty() && !model->ea_layer.t2d_map.empty());
}

llama_token llama_map_draft_to_target(const struct llama_model * model, llama_token draft_token) {
    if (!llama_model_has_eagle_vocab_mapping(model)) {
        return -1; // Invalid mapping
    }

    const int32_t *d2t_map = model->ea_layer.d2t_map.data();
    size_t d2t_size = model->ea_layer.d2t_map.size();

    if (draft_token >= 0 && draft_token < (llama_token)d2t_size) {
        int32_t diff = d2t_map[draft_token];
        int32_t target_token = diff + (int32_t)draft_token; // Apply diff transformation
        LLAMA_LOG_INFO("%s: draft_token=%d, diff=%d, target_token=%d, t2d_size=%zu\n",
                       __func__, (int)draft_token, diff, target_token, model->ea_layer.t2d_map.size());
        if (target_token >= 0 && target_token < (int32_t)model->ea_layer.t2d_map.size()) {
            return (llama_token)target_token;
        }
    }

    return -1; // Invalid mapping
}

bool llama_has_target_draft_mapping(const struct llama_model * model, llama_token target_token) {
    if (!llama_model_has_eagle_vocab_mapping(model)) {
        return false;
    }

    const uint8_t *t2d_map = model->ea_layer.t2d_map.data();
    size_t t2d_size = model->ea_layer.t2d_map.size();

    return (target_token >= 0 && target_token < (llama_token)t2d_size && t2d_map[target_token]);
}

bool llama_model_dflash_is_domino(const struct llama_model * model) {
    return model && model->arch == LLM_ARCH_DFLASH && model->dflash_domino;
}

uint32_t llama_model_dflash_domino_prefix_len(const struct llama_model * model) {
    return llama_model_dflash_is_domino(model) ? model->dflash_domino_prefix_len : 0;
}

bool llama_model_dflash_domino_shift_label(const struct llama_model * model) {
    return llama_model_dflash_is_domino(model) && model->dflash_domino_shift_label;
}

namespace {

struct dflash_domino_cache {
    std::vector<float> gru_w_ih;
    std::vector<float> gru_w_hh;
    std::vector<float> fc1;
    std::vector<float> fc2;
    int64_t hidden = 0;
    int64_t gru_hidden = 0;
    int64_t emb_dim = 0;
    int64_t vocab = 0;
};

struct dflash_domino_runtime_state {
    const llama_model * model_dft = nullptr;
    const llama_model * model_tgt = nullptr;
    std::vector<llama_token> prefix;
    std::vector<float> h;
    std::unordered_map<llama_token, std::vector<float>> emb_cache;
};

static void tensor_to_float(const ggml_tensor * t, std::vector<float> & out) {
    const int64_t n = ggml_nelements(t);
    out.resize(n);
    const size_t nbytes = ggml_nbytes(t);
    switch (t->type) {
        case GGML_TYPE_F32:
            ggml_backend_tensor_get(t, out.data(), 0, nbytes);
            break;
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> tmp(n);
            ggml_backend_tensor_get(t, tmp.data(), 0, nbytes);
            ggml_fp16_to_fp32_row(tmp.data(), out.data(), n);
        } break;
        case GGML_TYPE_BF16: {
            std::vector<ggml_bf16_t> tmp(n);
            ggml_backend_tensor_get(t, tmp.data(), 0, nbytes);
            ggml_bf16_to_fp32_row(tmp.data(), out.data(), n);
        } break;
        default:
            throw std::runtime_error("Domino tensors must be F32/F16/BF16");
    }
}

static void tensor_row_to_float_uncached(const ggml_tensor * t, int64_t row, std::vector<float> & out) {
    const int64_t cols = t->ne[0];
    out.resize(cols);
    const size_t offset = row * t->nb[1];
    const size_t nbytes = cols * ggml_type_size(t->type);
    switch (t->type) {
        case GGML_TYPE_F32:
            ggml_backend_tensor_get(t, out.data(), offset, nbytes);
            break;
        case GGML_TYPE_F16: {
            std::vector<ggml_fp16_t> tmp(cols);
            ggml_backend_tensor_get(t, tmp.data(), offset, nbytes);
            ggml_fp16_to_fp32_row(tmp.data(), out.data(), cols);
        } break;
        case GGML_TYPE_BF16: {
            std::vector<ggml_bf16_t> tmp(cols);
            ggml_backend_tensor_get(t, tmp.data(), offset, nbytes);
            ggml_bf16_to_fp32_row(tmp.data(), out.data(), cols);
        } break;
        default:
            throw std::runtime_error("Domino target embeddings must be F32/F16/BF16");
    }
}

static const std::vector<float> & tensor_row_to_float_cached(
        dflash_domino_runtime_state & state,
        const ggml_tensor * t,
        llama_token token) {
    auto it = state.emb_cache.find(token);
    if (it != state.emb_cache.end()) {
        return it->second;
    }

    std::vector<float> row;
    tensor_row_to_float_uncached(t, token, row);
    auto inserted = state.emb_cache.emplace(token, std::move(row));
    return inserted.first->second;
}

static const dflash_domino_cache & get_domino_cache(const llama_model * model) {
    static std::mutex mutex;
    static std::map<const llama_model *, std::unique_ptr<dflash_domino_cache>> caches;
    std::lock_guard<std::mutex> lock(mutex);
    auto it = caches.find(model);
    if (it != caches.end()) {
        return *it->second;
    }
    if (!llama_model_dflash_is_domino(model)) {
        throw std::runtime_error("DFlash model is not a Domino draft model");
    }

    auto cache = std::make_unique<dflash_domino_cache>();
    cache->hidden     = model->dflash_domino_gru_w_ih->ne[0];
    cache->gru_hidden = model->dflash_domino_gru_w_hh->ne[0];
    cache->emb_dim    = model->dflash_domino_fc1->ne[1];
    cache->vocab      = model->dflash_domino_fc2->ne[1];
    tensor_to_float(model->dflash_domino_gru_w_ih, cache->gru_w_ih);
    tensor_to_float(model->dflash_domino_gru_w_hh, cache->gru_w_hh);
    tensor_to_float(model->dflash_domino_fc1,      cache->fc1);
    tensor_to_float(model->dflash_domino_fc2,      cache->fc2);

    auto inserted = caches.emplace(model, std::move(cache));
    return *inserted.first->second;
}

static void gru_step(
        const dflash_domino_cache & c,
        const float * x,
        std::vector<float> & h,
        std::vector<float> & gi,
        std::vector<float> & gh) {
    const int64_t H = c.hidden;
    const int64_t G = c.gru_hidden;
    gi.assign(3 * G, 0.0f);
    gh.assign(3 * G, 0.0f);
    for (int64_t o = 0; o < 3 * G; ++o) {
        const float * w_ih = c.gru_w_ih.data() + o * H;
        const float * w_hh = c.gru_w_hh.data() + o * G;
        for (int64_t i = 0; i < H; ++i) {
            gi[o] += w_ih[i] * x[i];
        }
        for (int64_t i = 0; i < G; ++i) {
            gh[o] += w_hh[i] * h[i];
        }
    }

    std::vector<float> next(G);
    for (int64_t i = 0; i < G; ++i) {
        const float r = 1.0f / (1.0f + std::exp(-(gi[i] + gh[i])));
        const float z = 1.0f / (1.0f + std::exp(-(gi[G + i] + gh[G + i])));
        const float n = std::tanh(gi[2 * G + i] + r * gh[2 * G + i]);
        next[i] = (1.0f - z) * n + z * h[i];
    }
    h.swap(next);
}

static bool domino_prefix_matches(
        const std::vector<llama_token> & cached,
        const llama_token * prefix_tokens,
        int32_t n_prefix_tokens) {
    if (cached.size() != static_cast<size_t>(n_prefix_tokens)) {
        return false;
    }
    for (int32_t i = 0; i < n_prefix_tokens; ++i) {
        if (cached[i] != prefix_tokens[i]) {
            return false;
        }
    }
    return true;
}

static bool domino_prefix_is_one_token_extension(
        const std::vector<llama_token> & cached,
        const llama_token * prefix_tokens,
        int32_t n_prefix_tokens) {
    if (cached.size() + 1 != static_cast<size_t>(n_prefix_tokens)) {
        return false;
    }
    for (size_t i = 0; i < cached.size(); ++i) {
        if (cached[i] != prefix_tokens[i]) {
            return false;
        }
    }
    return true;
}

static void domino_update_prefix_state(
        const dflash_domino_cache & c,
        dflash_domino_runtime_state & state,
        const llama_model * model_dft,
        const llama_model * model_tgt,
        const llama_token * prefix_tokens,
        int32_t n_prefix_tokens) {
    if (state.model_dft != model_dft || state.model_tgt != model_tgt) {
        state.model_dft = model_dft;
        state.model_tgt = model_tgt;
        state.prefix.clear();
        state.h.clear();
        state.emb_cache.clear();
    }

    if (domino_prefix_matches(state.prefix, prefix_tokens, n_prefix_tokens)) {
        return;
    }

    std::vector<float> gi;
    std::vector<float> gh;

    if (domino_prefix_is_one_token_extension(state.prefix, prefix_tokens, n_prefix_tokens)) {
        const auto & emb = tensor_row_to_float_cached(state, model_tgt->tok_embd, prefix_tokens[n_prefix_tokens - 1]);
        if (state.h.empty()) {
            state.h.assign(c.gru_hidden, 0.0f);
        }
        gru_step(c, emb.data(), state.h, gi, gh);
        state.prefix.push_back(prefix_tokens[n_prefix_tokens - 1]);
        return;
    }

    state.prefix.assign(prefix_tokens, prefix_tokens + n_prefix_tokens);
    state.h.assign(c.gru_hidden, 0.0f);
    for (int32_t i = 0; i < n_prefix_tokens; ++i) {
        const auto & emb = tensor_row_to_float_cached(state, model_tgt->tok_embd, prefix_tokens[i]);
        gru_step(c, emb.data(), state.h, gi, gh);
    }
}

static int domino_thread_count() {
    static int n_threads = []() {
        if (const char * env = std::getenv("LLAMA_DOMINO_THREADS")) {
            try {
                return std::max(1, std::stoi(env));
            } catch (...) {
                return 1;
            }
        }
        const unsigned hw = std::thread::hardware_concurrency();
        return static_cast<int>(std::max(1u, std::min(hw == 0 ? 1u : hw, 8u)));
    }();
    return n_threads;
}

static std::pair<llama_token, float> domino_fc2_argmax(
        const dflash_domino_cache & c,
        const std::vector<float> & mid,
        const float * base_logits) {
    const int64_t V = c.vocab;
    const int64_t E = c.emb_dim;
    const int n_threads = std::min<int64_t>(domino_thread_count(), std::max<int64_t>(1, V));
    if (n_threads <= 1 || V < 4096) {
        llama_token best_id = 0;
        float best = -std::numeric_limits<float>::infinity();
        for (int64_t v = 0; v < V; ++v) {
            const float * w = c.fc2.data() + v * E;
            float logit = base_logits[v];
            for (int64_t i = 0; i < E; ++i) {
                logit += w[i] * mid[i];
            }
            if (logit > best) {
                best = logit;
                best_id = static_cast<llama_token>(v);
            }
        }
        return { best_id, best };
    }

    struct local_best {
        llama_token id = 0;
        float value = -std::numeric_limits<float>::infinity();
    };

    std::vector<local_best> bests(n_threads);
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    for (int t = 0; t < n_threads; ++t) {
        const int64_t begin = (V * t) / n_threads;
        const int64_t end   = (V * (t + 1)) / n_threads;
        workers.emplace_back([&, t, begin, end]() {
            llama_token best_id = static_cast<llama_token>(begin);
            float best = -std::numeric_limits<float>::infinity();
            for (int64_t v = begin; v < end; ++v) {
                const float * w = c.fc2.data() + v * E;
                float logit = base_logits[v];
                for (int64_t i = 0; i < E; ++i) {
                    logit += w[i] * mid[i];
                }
                if (logit > best) {
                    best = logit;
                    best_id = static_cast<llama_token>(v);
                }
            }
            bests[t] = { best_id, best };
        });
    }

    for (auto & worker : workers) {
        worker.join();
    }

    llama_token best_id = bests[0].id;
    float best = bests[0].value;
    for (int t = 1; t < n_threads; ++t) {
        if (bests[t].value > best) {
            best = bests[t].value;
            best_id = bests[t].id;
        }
    }
    return { best_id, best };
}

} // namespace

llama_token llama_dflash_domino_sample(
        const struct llama_context * ctx_dft,
        const struct llama_context * ctx_tgt,
        const llama_token * prefix_tokens,
        int32_t n_prefix_tokens,
        const float * parallel_hidden,
        const float * base_logits) {
    const llama_model * model_dft = llama_get_model(ctx_dft);
    const llama_model * model_tgt = llama_get_model(ctx_tgt);
    const dflash_domino_cache & c = get_domino_cache(model_dft);
    if (!model_tgt->tok_embd) {
        throw std::runtime_error("Domino requires target token embeddings");
    }

    thread_local dflash_domino_runtime_state state;
    domino_update_prefix_state(c, state, model_dft, model_tgt, prefix_tokens, n_prefix_tokens);

    std::vector<float> mid(c.emb_dim, 0.0f);
    for (int64_t o = 0; o < c.emb_dim; ++o) {
        const float * w = c.fc1.data() + o * (c.hidden + c.gru_hidden);
        float sum = 0.0f;
        for (int64_t i = 0; i < c.hidden; ++i) {
            sum += w[i] * parallel_hidden[i];
        }
        for (int64_t i = 0; i < c.gru_hidden; ++i) {
            sum += w[c.hidden + i] * state.h[i];
        }
        mid[o] = sum / (1.0f + std::exp(-sum));
    }

    return domino_fc2_argmax(c, mid, base_logits).first;
}

llama_token llama_dflash_domino_sample_gpu(
        struct llama_context * ctx_dft,
        struct llama_context * ctx_tgt,
        const llama_token * prefix_tokens,
        int32_t n_prefix_tokens,
        int32_t output_index,
        bool apply_correction) {
    const llama_model * model_dft = llama_get_model(ctx_dft);
    const llama_model * model_tgt = llama_get_model(ctx_tgt);

#ifdef GGML_USE_CUDA
    const bool domino_force_cpu = []() {
        const char * env = std::getenv("LLAMA_DOMINO_FORCE_CPU");
        return env && std::string(env) != "0";
    }();
    const ggml_tensor * target_output = model_tgt->output ? model_tgt->output : model_tgt->tok_embd;
    if (!domino_force_cpu && llama_model_dflash_is_domino(model_dft) && model_tgt->tok_embd && target_output) {
        ctx_dft->synchronize();
        llm_graph_result * res = ctx_dft->get_graph_result();
        ggml_tensor * t_hidden = res ? res->get_embd() : nullptr;
        if (t_hidden) {
            int32_t out_token = -1;
            const int rc = llama_cuda_domino_sample(
                    model_tgt->tok_embd,
                    target_output,
                    model_dft->dflash_domino_gru_w_ih,
                    model_dft->dflash_domino_gru_w_hh,
                    model_dft->dflash_domino_fc1,
                    model_dft->dflash_domino_fc2,
                    t_hidden,
                    output_index,
                    reinterpret_cast<const int32_t *>(prefix_tokens),
                    n_prefix_tokens,
                    apply_correction,
                    &out_token);
            if (rc == 0 && out_token >= 0) {
                return static_cast<llama_token>(out_token);
            }
            LLAMA_LOG_WARN("%s: CUDA Domino sample failed rc=%d, falling back to host path\n", __func__, rc);
        }
    }
#endif

    const float * parallel_hidden = llama_get_embeddings_ith(ctx_dft, output_index);
    const float * base_logits = llama_get_logits_ith(ctx_dft, output_index);
    GGML_ASSERT(parallel_hidden && "Domino DFlash hidden output missing");
    GGML_ASSERT(base_logits && "Domino DFlash base logits missing");
    return llama_dflash_domino_sample(
            ctx_dft,
            ctx_tgt,
            prefix_tokens,
            n_prefix_tokens,
            parallel_hidden,
            base_logits);
}


// 声明 ggml-cuda.cu 里的函数
extern "C" void llama_ssd_clear_cache(); 
extern "C" void llama_ssd_set_cuda_cache_mode(int mode);
extern "C" void llama_ssd_clear_cuda_cache();
extern "C" void llama_ssd_backend_profile_reset();
extern "C" void llama_ssd_backend_profile_snapshot(struct llama_ssd_profile * profile);

// 导出给用户的 API
void llama_clear_ssd_cache_backend() {
    // 调用后端清理
    // (需要确保编译了 CUDA 后端，否则这里可能会链接错误，可以用 #ifdef GGML_USE_CUDA)
#ifdef GGML_USE_CUDA
    llama_ssd_clear_cache();
#endif
}

void llama_set_ssd_cuda_cache_mode(int mode) {
#ifdef GGML_USE_CUDA
    llama_ssd_set_cuda_cache_mode(mode);
#else
    (void) mode;
#endif
}

void llama_clear_ssd_cuda_cache(void) {
#ifdef GGML_USE_CUDA
    llama_ssd_clear_cuda_cache();
#endif
}

void llama_ssd_profile_reset(void) {
#ifdef GGML_USE_CUDA
    llama_ssd_backend_profile_reset();
#endif
}

void llama_ssd_profile_snapshot(struct llama_ssd_profile * profile) {
    if (profile == nullptr) {
        return;
    }
#ifdef GGML_USE_CUDA
    llama_ssd_backend_profile_snapshot(profile);
#else
    *profile = {};
#endif
}
