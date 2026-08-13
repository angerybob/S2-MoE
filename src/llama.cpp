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
#include <regex>
#include <set>
#include <unistd.h>

// 复用之前定义的结构体 (最好放在头文件里，这里先复制一份以防万一)
// struct ssd_expert_info_t {
//     uint16_t file_idx;
//     size_t   file_offs;
//     size_t   size;
//     ggml_type type;
//     std::vector<int64_t> ne;
//     std::string fname; // 文件路径
// };

// 全局注册表：Key = "tensor_name"
// static std::map<std::string, ssd_expert_info_t> g_ssd_experts;
// static std::mutex g_ssd_mutex;

// 供外部调用的 C 风格接口 (将在 ggml-cuda.cu 中声明使用)
// extern "C" {
//     // 根据 Tensor 名和 Expert ID 获取偏移量和文件路径
//     // 如果找到了，返回 1，并填充 fname_out, offset_out, size_out
//     int llama_ssd_get_expert(const char* base_name, int expert_id, char* fname_out, size_t* offset_out, size_t* size_out) {
//         std::lock_guard<std::mutex> lock(g_ssd_mutex);
        
//         // base_name 类似于 "blk.0.ffn_gate.weight" (幽灵张量的名字)
//         // 我们需要把它转换成 "blk.0.ffn_gate.000.weight"
//         std::string name(base_name);
        
//         // 简单的字符串替换逻辑：插入 expert_id
//         // 假设 base_name 格式为 "blk.L.TYPE.weight" (exps被去掉了，或者依然是 ffn_gate_exps)
//         // 我们在 Step D 里把幽灵张量命名为了 "blk.X.ffn_gate_exps.weight"
//         // 我们的注册表里存的是 "blk.X.ffn_gate.000.weight"
        
//         char buf[256];
//         int layer_id = -1;
//         char type_str[64] = {0};
        
//         // 解析 Layer ID
//         if (sscanf(name.c_str(), "blk.%d.%[^.]", &layer_id, type_str) == 2) {
//             std::string type(type_str);
//             // 映射: ffn_gate_exps -> ffn_gate
//             if (type == "ffn_gate_exps") type = "ffn_gate";
//             if (type == "ffn_down_exps") type = "ffn_down";
//             if (type == "ffn_up_exps")   type = "ffn_up";
            
//             snprintf(buf, 256, "blk.%d.%s.%03d.weight", layer_id, type.c_str(), expert_id);
//         } else {
//             return 0; // 解析失败
//         }
        
//         std::string key(buf);
//         if (g_ssd_experts.find(key) != g_ssd_experts.end()) {
//             const auto& info = g_ssd_experts[key];
//             strcpy(fname_out, info.fname.c_str());
//             *offset_out = info.file_offs;
//             *size_out = info.size;
//             return 1;
//         }
//         return 0;
//     }
// }
// --- INSERT END ---

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif
extern "C" void ggml_cuda_register_ssd_expert(const char* key, const char* fname, size_t offset, size_t size, int resident);
extern "C" void ggml_cuda_print_ssd_expert_stats(void);
extern "C" void llama_ssd_clear_cache(void);
extern "C" void llama_ssd_set_cuda_cache_mode(int mode);
extern "C" void llama_ssd_clear_cuda_cache(void);
extern "C" void ggml_backend_moe_expert_capture_clear(void);
extern "C" int32_t ggml_backend_moe_expert_capture_get(int32_t * layers, int32_t * experts, int32_t max_items);

#ifdef GGML_USE_CUDA
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
#endif

static std::set<std::pair<int, int>> llama_load_gpu_experts_json(const char * path) {
    std::set<std::pair<int, int>> result;
    if (path == nullptr || path[0] == '\0') {
        return result;
    }

    std::ifstream in(path);
    if (!in) {
        LLAMA_LOG_WARN("%s: failed to open gpu experts json '%s'\n", __func__, path);
        return result;
    }

    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::regex pair_re("\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
    for (std::sregex_iterator it(text.begin(), text.end(), pair_re), end; it != end; ++it) {
        result.emplace(std::stoi((*it)[1].str()), std::stoi((*it)[2].str()));
    }

    LLAMA_LOG_INFO("%s: loaded %zu resident GPU expert ids from %s\n", __func__, result.size(), path);
    return result;
}

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

void llama_clear_ssd_cache_backend(void) {
    llama_ssd_clear_cache();
}

void llama_set_ssd_cuda_cache_mode(int mode) {
    llama_ssd_set_cuda_cache_mode(mode);
}

void llama_clear_ssd_cuda_cache(void) {
    llama_ssd_clear_cuda_cache();
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

void llama_moe_expert_capture_clear(void) {
    ggml_backend_moe_expert_capture_clear();
}

int32_t llama_moe_expert_capture_get(int32_t * layers, int32_t * experts, int32_t max_items) {
    return ggml_backend_moe_expert_capture_get(layers, experts, max_items);
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
        llama_model_loader ml(fname, splits, params.use_mmap, params.check_tensors, params.kv_overrides, params.tensor_buft_overrides);

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
        const auto gpu_experts = llama_load_gpu_experts_json(params.gpu_experts_json);
        // std::lock_guard<std::mutex> lock(g_ssd_mutex);
        // auto experts = model->get_ssd_experts();
        // for (auto const& [key, val] : experts) {
        //     ssd_expert_info_t info;
        //     info.file_idx = val.file_idx;
        //     info.file_offs = val.file_offs;
        //     info.size = val.size;
        //     info.type = val.type;
        //     info.ne = val.ne;
        //     // 找到对应的文件名
        //     // 这里我们需要访问 model->pimpl->mappings 里的 file 指针来获取文件名
        //     // 这是一个 hack，我们假设 splits[val.file_idx] 就是文件名
        //     // 如果 splits 为空（单文件），则用 path_model
        //     if (val.file_idx < splits.size()) {
        //         info.fname = splits[val.file_idx];
        //     } else {
        //         info.fname = path_model;
        //     }
        auto experts = model->get_ssd_experts(); // 使用 getter
        for (auto const& [key, val] : experts) {
            std::string fname;
            // 获取文件名逻辑 (简单的 hack，假设 file_idx 对应 splits)
            if (val.file_idx < splits.size()) {
                fname = splits[val.file_idx];
            } else {
                fname = path_model;
            }
            // g_ssd_experts[key] = info;
            int layer_id = -1;
            int expert_id = -1;
            char type_str[64] = {};
            const int matched = sscanf(key.c_str(), "blk.%d.%63[^.].%d.weight", &layer_id, type_str, &expert_id);
            const bool resident = matched == 3 && gpu_experts.count({layer_id, expert_id}) > 0;
            ggml_cuda_register_ssd_expert(key.c_str(), fname.c_str(), val.file_offs, val.size, resident ? 1 : 0);
        }
        // LLAMA_LOG_INFO("%s: Synced %zu SSD experts to global registry\n", __func__, g_ssd_experts.size());
        LLAMA_LOG_INFO("%s: Registered %zu SSD experts to CUDA backend\n", __func__, experts.size());
        ggml_cuda_print_ssd_expert_stats();
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

    if (hidden_states->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(hidden_states, output, offset, row_size);
    } else if (hidden_states->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n_embd);
        ggml_backend_tensor_get(hidden_states, tmp.data(), offset, row_size);
        ggml_fp16_to_fp32_row(tmp.data(), output, n_embd);
    } else if (hidden_states->type == GGML_TYPE_BF16) {
        std::vector<ggml_bf16_t> tmp(n_embd);
        ggml_backend_tensor_get(hidden_states, tmp.data(), offset, row_size);
        ggml_bf16_to_fp32_row(tmp.data(), output, n_embd);
    } else {
        const ggml_type_traits * traits = ggml_get_type_traits(hidden_states->type);
        if (traits == nullptr || traits->to_float == nullptr) {
            free(output);
            *out_n_embd = 0;
            return nullptr;
        }
        std::vector<uint8_t> tmp(row_size);
        ggml_backend_tensor_get(hidden_states, tmp.data(), offset, row_size);
        traits->to_float(tmp.data(), output, n_embd);
    }

    *out_n_embd = (int32_t)n_embd;
    return output;
}

// EAGLE3 SMP: Extract EAGLE hidden states for a specific token in the current batch
float * llama_context_get_eagle_hs_at_batch_idx(struct llama_context * ctx, int batch_idx, int32_t * out_dim) {
    if (!ctx || batch_idx < 0 || !out_dim) {
        return nullptr;
    }

    const struct llama_model * model = llama_get_model(ctx);
    const int32_t n_embd = model->hparams.n_embd;

    std::vector<int> eagle_layers;
    int32_t captured_count = 0;
    const int * captured = llama_context_get_hidden_states_layer_indices(ctx, &captured_count);
    if (captured != nullptr && captured_count > 0) {
        for (int32_t i = 0; i < captured_count; ++i) {
            if (std::find(eagle_layers.begin(), eagle_layers.end(), captured[i]) == eagle_layers.end()) {
                eagle_layers.push_back(captured[i]);
            }
        }
    } else {
        const int n_layer = model->hparams.n_layer;
        eagle_layers = {2, n_layer/2, n_layer-3};
    }
    const int num_layers = (int) eagle_layers.size();

    // **关键修复**：如果有最新的GPU结果，强制进行GPU->CPU同步
    auto * graph_result = ctx->get_graph_result();
    if (graph_result) {
        LLAMA_LOG_DEBUG("EAGLE3: graph_result captured_count=%d for batch_idx=%d\n", captured_count, batch_idx);
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
            LLAMA_LOG_ERROR("EAGLE3: missing hidden state layer %d for batch_idx=%d (captured_count=%d)\n",
                    layer_id, batch_idx, captured_count);
            free(output);
            return nullptr; // Layer not found
        }

        // Calculate offset: batch_idx * n_embd
        // Check bounds (use size_t to avoid integer overflow)
        if ((size_t)(batch_idx + 1) * n_embd > it->second.size()) {
            LLAMA_LOG_ERROR("EAGLE3: hidden state layer %d too small for batch_idx=%d: have=%zu floats need=%zu\n",
                    layer_id, batch_idx, it->second.size(), (size_t)(batch_idx + 1) * n_embd);
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
    LLAMA_LOG_DEBUG("EAGLE3: llama_context_set_target_hidden_states called - ctx=%p, hidden_states=%p, size=%zu\n",
                   (void*)ctx, (const void*)hidden_states, size);

    if (ctx && hidden_states) {
        // Log complete range for debugging - check all elements
        float hs_min = hidden_states[0], hs_max = hidden_states[0];
        for (size_t i = 1; i < size; i++) {
            hs_min = std::min(hs_min, hidden_states[i]);
            hs_max = std::max(hs_max, hidden_states[i]);
        }
        LLAMA_LOG_DEBUG("EAGLE3: hidden_states input stats - min: %.6f, max: %.6f (full range, %zu elements)\n", hs_min, hs_max, size);

        ctx->set_target_hidden_states(hidden_states, size);
        LLAMA_LOG_DEBUG("EAGLE3: set_target_hidden_states completed successfully\n");
    } else {
        LLAMA_LOG_DEBUG("EAGLE3: set_target_hidden_states failed - ctx=%p, hidden_states=%p\n",
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

void llama_context_set_moe_reuse_verify(
        struct llama_context * ctx,
        bool enabled,
        bool runtime_strength) {
    if (ctx) {
        ctx->set_moe_reuse_verify(enabled, runtime_strength);
    }
}

//
// EAGLE3: Draft model hidden states public API
//

void llama_context_set_draft_hidden_states(struct llama_context * ctx, const float * hidden_states, size_t size) {
    LLAMA_LOG_DEBUG("EAGLE3: llama_context_set_draft_hidden_states called - ctx=%p, hidden_states=%p, size=%zu\n",
                   (void*)ctx, (const void*)hidden_states, size);

    if (ctx && hidden_states) {
        // Log some sample values for debugging
        float hs_min = hidden_states[0], hs_max = hidden_states[0];
        for (int i = 1; i < std::min(int(size), 100); i++) {
            hs_min = std::min(hs_min, hidden_states[i]);
            hs_max = std::max(hs_max, hidden_states[i]);
        }
        LLAMA_LOG_DEBUG("EAGLE3: draft hidden_states input stats - min: %.6f, max: %.6f\n", hs_min, hs_max);

        ctx->set_draft_hidden_states(hidden_states, size);
        LLAMA_LOG_DEBUG("EAGLE3: set_draft_hidden_states completed successfully\n");
    } else {
        LLAMA_LOG_DEBUG("EAGLE3: set_draft_hidden_states failed - ctx=%p, hidden_states=%p\n",
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

void llama_set_eagle3_g_embeddings(struct llama_context * ctx, const float * g_embd, int32_t n_embd, int32_t n_tokens) {
    if (ctx) {
        ctx->set_eagle3_g_embeddings(g_embd, n_embd, n_tokens);
    }
}

const float * llama_get_eagle3_target_features(struct llama_context * ctx) {
    return ctx ? ctx->get_eagle3_target_features() : nullptr;
}

size_t llama_get_eagle3_target_features_size(struct llama_context * ctx) {
    return ctx ? ctx->get_eagle3_target_features_size() : 0;
}

void llama_select_eagle3_target_features(struct llama_context * ctx, const int32_t * indices, int32_t n_indices) {
    if (ctx) {
        ctx->select_eagle3_target_features(indices, n_indices);
    }
}

void llama_context_set_draft_context(struct llama_context * ctx, bool is_draft) {
    if (ctx) {
        ctx->set_draft_context(is_draft);
        LLAMA_LOG_DEBUG("EAGLE3: Set context %p as draft context: %s\n",
                       (void*)ctx, is_draft ? "true" : "false");
    }
}

bool llama_context_is_draft_context(const struct llama_context * ctx) {
    return ctx ? ctx->is_draft_ctx() : false;
}

void llama_context_set_target_embedding_layer(struct llama_context * ctx_draft, const struct llama_context * ctx_tgt) {
    LLAMA_LOG_DEBUG("EAGLE3: llama_context_set_target_embedding_layer called - draft_ctx=%p, target_ctx=%p\n",
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

    LLAMA_LOG_DEBUG("EAGLE3: Successfully set shared token embedding tensor - shape: [%lld, %lld]\n",
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
    LLAMA_LOG_DEBUG("EAGLE3: Cleared shared token embedding tensor\n");
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

void llama_model_dflash_set_target_model(
        struct llama_model * model,
        const struct llama_model * target_model) {
    if (!model || model->arch != LLM_ARCH_DFLASH || !target_model) {
        return;
    }
    model->target_tok_embd = target_model->tok_embd;
    model->target_output = target_model->output ? target_model->output : target_model->tok_embd;
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

static void tensor_row_to_float(const ggml_tensor * t, int64_t row, std::vector<float> & out) {
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

static const std::vector<float> & cached_embedding(
        dflash_domino_runtime_state & state,
        const ggml_tensor * t,
        llama_token token) {
    auto it = state.emb_cache.find(token);
    if (it != state.emb_cache.end()) {
        return it->second;
    }
    std::vector<float> row;
    tensor_row_to_float(t, token, row);
    return state.emb_cache.emplace(token, std::move(row)).first->second;
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
    tensor_to_float(model->dflash_domino_fc1, cache->fc1);
    tensor_to_float(model->dflash_domino_fc2, cache->fc2);
    return *caches.emplace(model, std::move(cache)).first->second;
}

static void gru_step(
        const dflash_domino_cache & c,
        const float * x,
        std::vector<float> & h) {
    const int64_t H = c.hidden;
    const int64_t G = c.gru_hidden;
    std::vector<float> gi(3 * G, 0.0f);
    std::vector<float> gh(3 * G, 0.0f);
    for (int64_t o = 0; o < 3 * G; ++o) {
        const float * w_ih = c.gru_w_ih.data() + o * H;
        const float * w_hh = c.gru_w_hh.data() + o * G;
        for (int64_t i = 0; i < H; ++i) gi[o] += w_ih[i] * x[i];
        for (int64_t i = 0; i < G; ++i) gh[o] += w_hh[i] * h[i];
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

static void update_prefix_state(
        const dflash_domino_cache & c,
        dflash_domino_runtime_state & state,
        const llama_model * model_dft,
        const llama_model * model_tgt,
        const llama_token * tokens,
        int32_t n_tokens) {
    bool same = state.model_dft == model_dft && state.model_tgt == model_tgt &&
        state.prefix.size() == static_cast<size_t>(n_tokens);
    if (same) {
        for (int32_t i = 0; i < n_tokens && same; ++i) same = state.prefix[i] == tokens[i];
    }
    if (same) return;

    bool extension = state.model_dft == model_dft && state.model_tgt == model_tgt &&
        state.prefix.size() + 1 == static_cast<size_t>(n_tokens);
    for (size_t i = 0; i < state.prefix.size() && extension; ++i) extension = state.prefix[i] == tokens[i];
    if (!extension) {
        state.model_dft = model_dft;
        state.model_tgt = model_tgt;
        state.prefix.clear();
        state.h.assign(c.gru_hidden, 0.0f);
        state.emb_cache.clear();
    }
    const size_t begin = state.prefix.size();
    for (size_t i = begin; i < static_cast<size_t>(n_tokens); ++i) {
        const auto & emb = cached_embedding(state, model_tgt->tok_embd, tokens[i]);
        gru_step(c, emb.data(), state.h);
        state.prefix.push_back(tokens[i]);
    }
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
    if (!model_tgt->tok_embd) throw std::runtime_error("Domino requires target token embeddings");

    thread_local dflash_domino_runtime_state state;
    update_prefix_state(c, state, model_dft, model_tgt, prefix_tokens, n_prefix_tokens);

    std::vector<float> mid(c.emb_dim, 0.0f);
    for (int64_t o = 0; o < c.emb_dim; ++o) {
        const float * w = c.fc1.data() + o * (c.hidden + c.gru_hidden);
        float sum = 0.0f;
        for (int64_t i = 0; i < c.hidden; ++i) sum += w[i] * parallel_hidden[i];
        for (int64_t i = 0; i < c.gru_hidden; ++i) sum += w[c.hidden + i] * state.h[i];
        mid[o] = sum / (1.0f + std::exp(-sum));
    }

    llama_token best_id = 0;
    float best = -std::numeric_limits<float>::infinity();
    for (int64_t v = 0; v < c.vocab; ++v) {
        const float * w = c.fc2.data() + v * c.emb_dim;
        float logit = base_logits[v];
        for (int64_t i = 0; i < c.emb_dim; ++i) logit += w[i] * mid[i];
        if (logit > best) { best = logit; best_id = static_cast<llama_token>(v); }
    }
    return best_id;
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
    const bool force_cpu = std::getenv("LLAMA_DOMINO_FORCE_CPU") != nullptr;
    const ggml_tensor * target_output = model_tgt->output ? model_tgt->output : model_tgt->tok_embd;
    if (!force_cpu && model_tgt->tok_embd && target_output) {
        ctx_dft->synchronize();
        llm_graph_result * res = ctx_dft->get_graph_result();
        ggml_tensor * hidden = res ? res->get_embd() : nullptr;
        if (hidden) {
            int32_t token = -1;
            const int rc = llama_cuda_domino_sample(
                model_tgt->tok_embd, target_output,
                model_dft->dflash_domino_gru_w_ih, model_dft->dflash_domino_gru_w_hh,
                model_dft->dflash_domino_fc1, model_dft->dflash_domino_fc2,
                hidden, output_index, reinterpret_cast<const int32_t *>(prefix_tokens),
                n_prefix_tokens, apply_correction, &token);
            if (rc == 0 && token >= 0) return static_cast<llama_token>(token);
        }
    }
#else
    GGML_UNUSED(apply_correction);
#endif
    const float * hidden = llama_get_embeddings_ith(ctx_dft, output_index);
    const float * logits = llama_get_logits_ith(ctx_dft, output_index);
    GGML_ASSERT(hidden && logits);
    return llama_dflash_domino_sample(ctx_dft, ctx_tgt, prefix_tokens, n_prefix_tokens, hidden, logits);
}
