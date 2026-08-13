#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "prune_routing_noise.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <chrono>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include "chat.h" // [新增]
#include <fstream>
#include <iostream>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>


// ================= Trace 结构体定义 =================
struct TokenLogit {
    int id;
    float val;
};

struct ExpertActivation {
    int layer;
    std::vector<int> expert_ids;
    std::vector<float> scores;
};

struct TraceNode {
    int node_id;
    int parent_id;
    int token_id;
    std::string token_str;
    int depth;
    
    // Metrics
    float confidence; // c_t
    float cost;       // delta_c (estimated bandwidth cost)
    bool accepted;
    
    // Experts
    std::vector<ExpertActivation> draft_experts;
    std::vector<ExpertActivation> predicted_experts;
    std::vector<ExpertActivation> target_experts;
    
    // Output Logits (只存 TopK 以节省空间)
    std::vector<TokenLogit> draft_logits_topk;
    std::vector<TokenLogit> target_logits_topk;
    int tmp_target_batch_idx;
};

struct StepTrace {
    int step_idx = 0;
    double t_target_verify_us = 0.0; // Target Model 验证这一批的耗时
    double t_target_ssd_fetch_us = 0.0;
    uint64_t target_ssd_read_requests = 0;
    uint64_t target_ssd_read_bytes = 0;
    int generated_tokens = 0;
    
    // 统计数据
    float total_cost = 0.0f;     // sum(delta_c)
    float effective_cost = 0.0f; // sum(delta_c of accepted)
    float redundant_cost = 0.0f; // sum(delta_c of rejected)
    
    std::vector<TraceNode> nodes;
};

// 全局 Trace 存储
static std::vector<StepTrace> g_all_traces;
static std::string g_trace_filename = "trace_output.json";

struct LayerProfileRow {
    int step = 0;
    int layer = -1;
    double elapsed_us = 0.0;
};

struct LayerProfileState {
    bool configured = false;
    bool active = false;
    int step = 0;
    int64_t last_us = 0;
    std::string filename;
    std::vector<LayerProfileRow> rows;
};

static LayerProfileState g_layer_profile;

static int64_t profile_now_us() {
    return (int64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool parse_layer_output_name(const char * name, int & layer) {
    static const char prefix[] = "l_out-";
    if (!name || strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return false;
    }
    char * end = nullptr;
    long value = strtol(name + sizeof(prefix) - 1, &end, 10);
    if (!end || *end != '\0' || value < 0) {
        return false;
    }
    layer = (int) value;
    return true;
}

static bool layer_profile_cb(ggml_tensor * t, bool ask, void * user_data) {
    (void) user_data;
    if (!g_layer_profile.configured || !g_layer_profile.active || !t) {
        return false;
    }
    int layer = -1;
    if (!parse_layer_output_name(t->name, layer)) {
        return false;
    }
    if (ask) {
        return true;
    }
    const int64_t now = profile_now_us();
    if (g_layer_profile.last_us > 0) {
        g_layer_profile.rows.push_back({
            g_layer_profile.step,
            layer,
            (double) (now - g_layer_profile.last_us),
        });
    }
    g_layer_profile.last_us = now;
    return true;
}

static void layer_profile_begin_step(int step) {
    if (!g_layer_profile.configured) {
        return;
    }
    g_layer_profile.step = step;
    g_layer_profile.active = true;
    g_layer_profile.last_us = profile_now_us();
}

static void layer_profile_end_step() {
    if (!g_layer_profile.configured) {
        return;
    }
    g_layer_profile.active = false;
    g_layer_profile.last_us = 0;
}

static void save_layer_profile_json() {
    if (!g_layer_profile.configured || g_layer_profile.filename.empty()) {
        return;
    }
    std::ofstream out(g_layer_profile.filename);
    out << "[\n";
    for (size_t i = 0; i < g_layer_profile.rows.size(); ++i) {
        const auto & row = g_layer_profile.rows[i];
        out << "  {\"step\":" << row.step
            << ",\"layer\":" << row.layer
            << ",\"elapsed_us\":" << row.elapsed_us
            << "}" << (i + 1 < g_layer_profile.rows.size() ? "," : "") << "\n";
    }
    out << "]\n";
    LOG_INF("Layer profile saved to %s (%zu rows)\n",
            g_layer_profile.filename.c_str(), g_layer_profile.rows.size());
}
static int g_capture_expert_count = 0;

// 辅助函数：获取 logits 的 TopK
static std::vector<TokenLogit> get_logits_topk(llama_context* ctx, int idx_in_batch, int k = 10) {
    auto * logits = llama_get_logits_ith(ctx, idx_in_batch);
    if (!logits) {
        return {};
    }
    // [修正] 正确获取词表大小的方式
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab); 
    
    std::vector<std::pair<float, int>> pairs;
    pairs.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        pairs.push_back({logits[i], i});
    }
    // Partial sort
    if (k > n_vocab) k = n_vocab;
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(), std::greater<std::pair<float, int>>());
    
    std::vector<TokenLogit> result;
    for (int i = 0; i < k; ++i) {
        result.push_back({pairs[i].second, pairs[i].first});
    }
    return result;
}

// 辅助函数：获取 MoE 激活情况 (封装 get_moe_info)
static std::vector<ExpertActivation> capture_experts(llama_context* ctx, int idx_in_batch) {
    const llama_model* model = llama_get_model(ctx);
    int n_layers = llama_model_n_layer(model);
    std::vector<llama_moe_topk_layer> layers(n_layers);
    int n_out_layers = 0;
    
    std::vector<ExpertActivation> result;
    
    // 调用 llama.cpp 的 API 获取激活信息
    // 注意：需要确保你的 llama.cpp 版本支持 llama_get_last_moe_topk
    if (llama_get_last_moe_topk(ctx, idx_in_batch, layers.data(), n_layers, &n_out_layers) && n_out_layers > 0) {
        for (int i = 0; i < n_out_layers; ++i) {
            ExpertActivation act;
            act.layer = layers[i].layer;
            for (int j = 0; j < layers[i].n; ++j) {
                const int expert_id = layers[i].expert_id[j];
                if (expert_id < 0 ||
                    (g_capture_expert_count > 0 &&
                     expert_id >= g_capture_expert_count)) {
                    continue;
                }
                act.expert_ids.push_back(expert_id);
                act.scores.push_back(layers[i].score[j]);
            }
            if (!act.expert_ids.empty()) {
                result.push_back(std::move(act));
            }
        }
    }
    return result;
}

struct RoutingReuseStats {
    long long verify_steps = 0;
    long long target_activated_experts = 0;
    long long target_unique_experts = 0;
    long long target_batch_reuse_hits = 0;
    long long target_new_routed_experts = 0;
    long long target_resident_routed_hits = 0;
    long long target_offchip_routed_experts = 0;
};

static std::set<std::pair<int, int>> load_hot_expert_set(const std::string & path) {
    std::set<std::pair<int, int>> result;
    if (path.empty()) {
        return result;
    }
    std::ifstream in(path);
    if (!in) {
        return result;
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    size_t pos = 0;
    while (true) {
        pos = text.find('[', pos);
        if (pos == std::string::npos) {
            break;
        }
        size_t end = text.find(']', pos + 1);
        if (end == std::string::npos) {
            break;
        }
        const std::string item = text.substr(pos + 1, end - pos - 1);
        int layer = -1;
        int expert = -1;
        if (std::sscanf(item.c_str(), " %d , %d ", &layer, &expert) == 2) {
            result.insert({layer, expert});
        }
        pos = end + 1;
    }
    return result;
}

static void update_target_routing_reuse_stats(
        llama_context * ctx,
        int n_tokens_in_batch,
        const std::set<std::pair<int, int>> & resident_experts,
        std::set<std::pair<int, int>> & seen_experts,
        RoutingReuseStats & stats) {
    std::set<std::pair<int, int>> batch_unique;
    long long activated = 0;
    for (int token_idx = 0; token_idx < n_tokens_in_batch; ++token_idx) {
        const auto experts = capture_experts(ctx, token_idx);
        for (const auto & layer : experts) {
            for (const int expert_id : layer.expert_ids) {
                activated++;
                batch_unique.insert({layer.layer, expert_id});
            }
        }
    }
    if (activated == 0) {
        return;
    }
    long long resident_hits = 0;
    long long offchip = 0;
    long long new_routed = 0;
    for (const auto & key : batch_unique) {
        if (resident_experts.find(key) != resident_experts.end()) {
            resident_hits++;
        } else {
            offchip++;
        }
        if (seen_experts.insert(key).second) {
            new_routed++;
        }
    }
    stats.verify_steps++;
    stats.target_activated_experts += activated;
    stats.target_unique_experts += (long long) batch_unique.size();
    stats.target_batch_reuse_hits += activated - (long long) batch_unique.size();
    stats.target_new_routed_experts += new_routed;
    stats.target_resident_routed_hits += resident_hits;
    stats.target_offchip_routed_experts += offchip;
}

// JSON 导出函数
static void save_trace_to_json(const std::string& filename) {
    std::ofstream out(filename);
    out << "[\n";
    for (size_t i = 0; i < g_all_traces.size(); ++i) {
        const auto& step = g_all_traces[i];
        out << "  {\n";
        out << "    \"step\": " << step.step_idx << ",\n";
        out << "    \"t_verify_us\": " << step.t_target_verify_us << ",\n";
        out << "    \"t_target_ssd_fetch_us\": " << step.t_target_ssd_fetch_us << ",\n";
        out << "    \"target_ssd_read_requests\": " << step.target_ssd_read_requests << ",\n";
        out << "    \"target_ssd_read_bytes\": " << step.target_ssd_read_bytes << ",\n";
        out << "    \"generated_tokens\": " << step.generated_tokens << ",\n";
        out << "    \"cost_total\": " << step.total_cost << ",\n";
        out << "    \"cost_effective\": " << step.effective_cost << ",\n";
        out << "    \"cost_redundant\": " << step.redundant_cost << ",\n";
        out << "    \"nodes\": [\n";
        
        for (size_t j = 0; j < step.nodes.size(); ++j) {
            const auto& node = step.nodes[j];
            out << "      {\n";
            out << "        \"id\": " << node.node_id << ",\n";
            out << "        \"parent\": " << node.parent_id << ",\n";
            out << "        \"token\": " << node.token_id << ",\n";
            // 简单转义 token 字符串
            std::string safe_str;
            for(char c : node.token_str) {
                if(c == '"') safe_str += "\\\"";
                else if(c == '\\') safe_str += "\\\\";
                else if(c >= 0x20 && c <= 0x7E) safe_str += c;
            }
            out << "        \"str\": \"" << safe_str << "\",\n";
            out << "        \"depth\": " << node.depth << ",\n";
            out << "        \"conf\": " << node.confidence << ",\n";
            out << "        \"cost\": " << node.cost << ",\n";
            out << "        \"accepted\": " << (node.accepted ? "true" : "false") << ",\n";
            
            // Draft Experts
            out << "        \"draft_experts\": [";
            for(size_t k=0; k<node.draft_experts.size(); ++k) {
                out << "{\"l\":" << node.draft_experts[k].layer << ",\"ids\":[";
                for(size_t m=0; m<node.draft_experts[k].expert_ids.size(); ++m) 
                    out << node.draft_experts[k].expert_ids[m] << (m+1<node.draft_experts[k].expert_ids.size()?",":"");
                out << "],\"scores\":[";
                for(size_t m=0; m<node.draft_experts[k].scores.size(); ++m)
                    out << node.draft_experts[k].scores[m] << (m+1<node.draft_experts[k].scores.size()?",":"");
                out << "]}";
                if(k+1 < node.draft_experts.size()) out << ",";
            }
            out << "],\n";

            out << "        \"predicted_experts\": [";
            for(size_t k=0; k<node.predicted_experts.size(); ++k) {
                out << "{\"l\":" << node.predicted_experts[k].layer << ",\"ids\":[";
                for(size_t m=0; m<node.predicted_experts[k].expert_ids.size(); ++m)
                    out << node.predicted_experts[k].expert_ids[m] << (m+1<node.predicted_experts[k].expert_ids.size()?",":"");
                out << "],\"scores\":[";
                for(size_t m=0; m<node.predicted_experts[k].scores.size(); ++m)
                    out << node.predicted_experts[k].scores[m] << (m+1<node.predicted_experts[k].scores.size()?",":"");
                out << "]}";
                if(k+1 < node.predicted_experts.size()) out << ",";
            }
            out << "],\n";

            // Target Experts
            out << "        \"target_experts\": [";
            for(size_t k=0; k<node.target_experts.size(); ++k) {
                out << "{\"l\":" << node.target_experts[k].layer << ",\"ids\":[";
                for(size_t m=0; m<node.target_experts[k].expert_ids.size(); ++m) 
                    out << node.target_experts[k].expert_ids[m] << (m+1<node.target_experts[k].expert_ids.size()?",":"");
                out << "],\"scores\":[";
                for(size_t m=0; m<node.target_experts[k].scores.size(); ++m)
                    out << node.target_experts[k].scores[m] << (m+1<node.target_experts[k].scores.size()?",":"");
                out << "]}";
                if(k+1 < node.target_experts.size()) out << ",";
            }
            out << "]\n"; // End of experts
            
            // 可以按需添加 logits 输出，这里为了文件大小先省略，如果需要请仿照 experts 写法
            
            out << "      }" << (j + 1 < step.nodes.size() ? "," : "") << "\n";
        }
        out << "    ]\n";
        out << "  }" << (i + 1 < g_all_traces.size() ? "," : "") << "\n";
    }
    out << "]\n";
    out.close();
    LOG_INF("Trace saved to %s\n", filename.c_str());
}
// ====================================================
#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// static const float PRUNE_BUDGET_B = 145.0f;
// static const int PRUNE_M_ROUTE = 8;
// static const int PRUNE_K_TGT = 8;
// static const float PRUNE_BETA = 0.8f;
// static const float PRUNE_GAMMA = 2.0f;
// static const float PRUNE_LAMBDA = 1.0f;
// static const float PRUNE_TPOT = 30.0f;
// static const float PRUNE_EPS = 9.0f;
// static const float PRUNE_EXPERT_BYTES = 32.768f;
// static const float PRUNE_BANDWIDTH = 204.8f;
// static const int PRUNE_EXPERT_MAX = 384;
// =================================================================================================
// [新增] 定义全局配置对象，用于在 main 函数之外访问参数
// =================================================================================================
struct GlobalPruneConfig {
    float budget_b      = 145.0f;
    int   m_route       = 8;
    int   k_tgt         = 8;
    float beta          = 0.8f;
    float gamma         = 2.0f;
    float lambda        = 1.0f;
    float tpot          = 30.0f;
    float eps           = 9.0f;
    float expert_bytes  = 32.768f;
    float bandwidth     = 204.8f;
    int   expert_max    = 384;
    float routing_noise = 0.0f;
    int   routing_noise_seed = 1;
    int   routing_noise_experts = 0;
    bool  enable_trace  = false; 
};

// 声明全局实例，并初始化为默认值
static GlobalPruneConfig g_prune_config;

// MoE utility-driven draft length (Cascade-style: periodic test/set, arXiv:2506.20675)
struct MoeUtilityCascadeState {
    bool enabled = false;
    int cap = 0;
    std::vector<int> ks;
    bool testing = true;
    int test_idx = 0;
    float test_sum = 0.f;
    int test_n = 0;
    int test_iters = 4;
    int set_iters_base = 16;
    int set_left = 0;
    int set_mul = 1;
    int best_k = 0;
    float best_avg = -1e30f;
    int64_t pending_v_us = 1;
    bool has_pending = false;

    void init(const common_params_speculative & sp) {
        enabled = sp.moe_utility_spec;
        if (!enabled) {
            return;
        }
        cap = std::max(0, (int) sp.n_max);
        test_iters = std::max(1, (int) sp.utility_test_iters);
        set_iters_base = std::max(1, (int) sp.utility_set_iters);
        std::set<int> cand;
        cand.insert(0);
        // if (cap >= 1) {
        //     cand.insert(1);
        // }
        // if (cap >= 2) {
        //     const int mid = std::max(2, cap / 2);
        //     if (mid <= cap) {
        //         cand.insert(mid);
        //     }
        // }
        cand.insert(cap);
        ks.assign(cand.begin(), cand.end());
        testing = true;
        test_idx = 0;
        test_sum = 0.f;
        test_n = 0;
        best_avg = -1e30f;
        best_k = cap;
        has_pending = false;
        set_mul = 1;
        set_left = 0;
    }

    int active_k() const {
        if (!enabled) {
            return cap;
        }
        if (testing) {
            return ks[std::min(test_idx, (int) ks.size() - 1)];
        }
        return best_k;
    }

    void on_verify_done(int64_t v_us) {
        if (!enabled) {
            return;
        }
        pending_v_us = std::max<int64_t>(1, v_us);
        has_pending = true;
    }

    void on_outer_iter_start(int n_predict, int & n_predict_mark) {
        if (!enabled) {
            n_predict_mark = n_predict;
            return;
        }
        if (has_pending) {
            const int dtok = n_predict - n_predict_mark;
            const float util = float(dtok) / float(pending_v_us);
            has_pending = false;

            if (testing) {
                test_sum += util;
                test_n++;
                if (test_n >= test_iters) {
                    const float avg = test_sum / float(test_n);
                    if (avg > best_avg) {
                        best_avg = avg;
                        best_k = ks[test_idx];
                    }
                    test_sum = 0.f;
                    test_n = 0;
                    test_idx++;
                    if (test_idx >= (int) ks.size()) {
                        testing = false;
                        set_mul = (best_k == 0) ? std::min(set_mul * 2, 8) : 1;
                        set_left = set_iters_base * set_mul;
                        test_idx = 0;
                        LOG_INF("%s: moe-utility-spec: test round done, best_k=%d best_avg_tokens_per_us=%e\n",
                                __func__, best_k, (double) best_avg);
                    }
                }
            } else {
                set_left--;
                if (set_left <= 0) {
                    testing = true;
                    test_idx = 0;
                    test_sum = 0.f;
                    test_n = 0;
                    best_avg = -1e30f;
                    best_k = cap;
                    LOG_INF("%s: moe-utility-spec: restarting test phase (cap=%d)\n", __func__, cap);
                }
            }
        }
        n_predict_mark = n_predict;
    }
};

struct NvtxRange {
    nvtxRangeId_t id;
    NvtxRange(const char * name) { id = nvtxRangeStartA(name); }
    ~NvtxRange() { nvtxRangeEnd(id); }
};


struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    // ================= [新增] =================
    std::vector<float> probs;  // 存储每个 token 的原始概率 P(x|ctx)
    // ==========================================
    std::vector<std::vector<llama_token_data>> dists;
    std::vector<std::vector<float>> moe_layer_ratios;
    std::vector<float> moe_ratio_avg;
    std::vector<std::vector<int>> moe_layer_counts;

    struct common_sampler * smpl = nullptr;
};

static void draft_moe_push(seq_draft & draft, const std::vector<float> & layer_ratios, const std::vector<int> & layer_counts, float avg_ratio) {
    draft.moe_layer_ratios.push_back(layer_ratios);
    draft.moe_ratio_avg.push_back(avg_ratio);
    draft.moe_layer_counts.push_back(layer_counts);
}

static void draft_moe_push_empty(seq_draft & draft) {
    draft.moe_layer_ratios.emplace_back();
    draft.moe_ratio_avg.push_back(-1.0f);
    draft.moe_layer_counts.emplace_back();
}

struct ExpertSet {
    std::vector<uint64_t> bits;

    ExpertSet() : bits((g_prune_config.expert_max + 63) / 64, 0) {}

    void clear() {
        std::fill(bits.begin(), bits.end(), 0);
    }

    void set(int id) {
        if (id < 0) {
            return;
        }
        if (id >= g_prune_config.expert_max) {
            g_prune_config.expert_max = id + 1;
        }
        const size_t need = (size_t) id / 64 + 1;
        if (bits.size() < need) {
            bits.resize(need, 0);
        }
        bits[id / 64] |= (uint64_t(1) << (id % 64));
    }

    void or_with(const ExpertSet & other) {
        for (size_t i = 0; i < bits.size(); ++i) {
            bits[i] |= other.bits[i];
        }
    }

    void andnot_with(const ExpertSet & other) {
        for (size_t i = 0; i < bits.size(); ++i) {
            bits[i] &= ~other.bits[i];
        }
    }

    int popcount() const {
        int count = 0;
        for (uint64_t v : bits) {
            count += __builtin_popcountll(v);
        }
        return count;
    }
};

struct ExpertLayers {
    std::vector<ExpertSet> layers;

    ExpertLayers() = default;
    explicit ExpertLayers(int n_layers) { resize(n_layers); }

    void resize(int n_layers) {
        layers.assign(std::max(0, n_layers), ExpertSet());
    }

    void clear() {
        for (auto & layer : layers) {
            layer.clear();
        }
    }

    void set(int layer, int expert_id) {
        if (layer < 0 || layer >= (int) layers.size()) {
            return;
        }
        layers[layer].set(expert_id);
    }

    void or_with(const ExpertLayers & other) {
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].or_with(other.layers[i]);
        }
    }

    void andnot_with(const ExpertLayers & other) {
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i].andnot_with(other.layers[i]);
        }
    }

    int popcount() const {
        int count = 0;
        for (const auto & layer : layers) {
            count += layer.popcount();
        }
        return count;
    }
};

static std::vector<std::set<int>> expert_layers_to_sets(const ExpertLayers & experts) {
    std::vector<std::set<int>> result(experts.layers.size());
    for (size_t layer = 0; layer < experts.layers.size(); ++layer) {
        const auto & bits = experts.layers[layer].bits;
        for (size_t word = 0; word < bits.size(); ++word) {
            uint64_t value = bits[word];
            while (value != 0) {
                const int bit = __builtin_ctzll(value);
                result[layer].insert(static_cast<int>(word * 64 + bit));
                value &= value - 1;
            }
        }
    }
    return result;
}

static ExpertLayers expert_sets_to_layers(const std::vector<std::set<int>> & experts) {
    ExpertLayers result(static_cast<int>(experts.size()));
    for (size_t layer = 0; layer < experts.size(); ++layer) {
        for (const int expert : experts[layer]) {
            result.set(static_cast<int>(layer), expert);
        }
    }
    return result;
}

static std::vector<ExpertActivation> expert_layers_to_activations(const ExpertLayers & experts) {
    std::vector<ExpertActivation> result;
    const auto layers = expert_layers_to_sets(experts);
    for (size_t layer = 0; layer < layers.size(); ++layer) {
        if (layers[layer].empty()) {
            continue;
        }
        ExpertActivation activation;
        activation.layer = static_cast<int>(layer);
        activation.expert_ids.assign(layers[layer].begin(), layers[layer].end());
        result.push_back(std::move(activation));
    }
    return result;
}

static ExpertLayers perturb_predicted_experts(
        const ExpertLayers & experts,
        uint64_t seed) {
    if (g_prune_config.routing_noise <= 0.0f ||
        g_prune_config.routing_noise_experts <= 0) {
        return experts;
    }
    return expert_sets_to_layers(perturb_layered_expert_ids(
        expert_layers_to_sets(experts),
        g_prune_config.routing_noise,
        g_prune_config.routing_noise_experts,
        seed));
}

static float clamp01(float v) {
    if (v < 0.0f) {
        return 0.0f;
    }
    if (v > 1.0f) {
        return 1.0f;
    }
    return v;
}

// ================= [修改] 通用 JSON 字符串提取工具 =================
// 支持 "key": "value" 和 "key": ["value"] (取列表第一个字符串)
static std::string extract_json_value(const std::string& json_line, const std::string& key) {
    std::string search_key = "\"" + key + "\"";
    size_t key_pos = json_line.find(search_key);
    if (key_pos == std::string::npos) return "";

    // 从 key 结束的位置开始找冒号
    size_t colon_pos = json_line.find(':', key_pos + search_key.length());
    if (colon_pos == std::string::npos) return "";

    // 从冒号后面开始找第一个双引号 (这就自动跳过了可能的 '[' 空格等字符)
    size_t start_quote = json_line.find('"', colon_pos + 1);
    if (start_quote == std::string::npos) return "";

    std::string result;
    bool escape = false;
    
    // 逐字符读取，直到遇到未转义的结束引号
    for (size_t i = start_quote + 1; i < json_line.length(); ++i) {
        char c = json_line[i];
        
        if (escape) {
            // 处理转义字符
            switch (c) {
                case 'n': result += '\n'; break;
                case 'r': result += '\r'; break;
                case 't': result += '\t'; break;
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                default: result += c; break;
            }
            escape = false;
        } else {
            if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                // 找到结束引号，停止
                return result; 
            } else {
                result += c;
            }
        }
    }
    return ""; // 格式错误，没找到结束引号
}
// ==============================================================

static bool extract_json_int_value(const std::string & json_line, const std::string & key, int32_t & value_out) {
    std::string search_key = "\"" + key + "\"";
    size_t key_pos = json_line.find(search_key);
    if (key_pos == std::string::npos) {
        return false;
    }
    size_t colon_pos = json_line.find(':', key_pos + search_key.length());
    if (colon_pos == std::string::npos) {
        return false;
    }
    size_t pos = colon_pos + 1;
    while (pos < json_line.size() && std::isspace(static_cast<unsigned char>(json_line[pos]))) {
        ++pos;
    }
    if (pos >= json_line.size()) {
        return false;
    }
    size_t end = pos;
    if (json_line[end] == '-') {
        ++end;
    }
    while (end < json_line.size() && std::isdigit(static_cast<unsigned char>(json_line[end]))) {
        ++end;
    }
    if (end == pos || (end == pos + 1 && json_line[pos] == '-')) {
        return false;
    }
    value_out = std::stoi(json_line.substr(pos, end - pos));
    return true;
}

static bool extract_json_int_array(const std::string & json_line, const std::string & key, std::vector<int32_t> & out) {
    std::string search_key = "\"" + key + "\"";
    size_t key_pos = json_line.find(search_key);
    if (key_pos == std::string::npos) {
        return false;
    }
    size_t colon_pos = json_line.find(':', key_pos + search_key.length());
    if (colon_pos == std::string::npos) {
        return false;
    }
    size_t start = json_line.find('[', colon_pos);
    if (start == std::string::npos) {
        return false;
    }
    size_t end = json_line.find(']', start);
    if (end == std::string::npos || end <= start) {
        return false;
    }
    out.clear();
    std::string body = json_line.substr(start + 1, end - start - 1);
    std::stringstream ss(body);
    std::string item;
    auto trim = [](std::string & s) {
        size_t b = s.find_first_not_of(" \t\r\n");
        size_t e = s.find_last_not_of(" \t\r\n");
        if (b == std::string::npos) {
            s.clear();
            return;
        }
        s = s.substr(b, e - b + 1);
    };
    while (std::getline(ss, item, ',')) {
        trim(item);
        if (item.empty()) {
            continue;
        }
        out.push_back(std::stoi(item));
    }
    return true;
}

static float compute_alpha(float c_t, float tae) {
    const float c = clamp01(c_t);
    const float base = std::pow(c, g_prune_config.lambda);
    const float atten = std::exp(-g_prune_config.beta * tae * std::pow(1.0f - c, g_prune_config.gamma));
    // return clamp01(base * atten);
    return base;
}

static void debug_dump_logits_if_nan(
    llama_context * ctx,
    const llama_model * model,
    int token_index,
    const char * tag) {
    float * logits = llama_get_logits_ith(ctx, token_index);
    if (!logits) {
        fprintf(stderr, "[SPEC DEBUG] %s: logits ptr null (idx=%d)\n", tag, token_index);
        return;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    int nan_count = 0;
    int inf_count = 0;
    int first_nan = -1;
    float min_v = 0.0f;
    float max_v = 0.0f;
    bool minmax_init = false;
    for (int i = 0; i < n_vocab; ++i) {
        const float v = logits[i];
        if (!std::isfinite(v)) {
            if (std::isnan(v)) {
                nan_count++;
                if (first_nan == -1) first_nan = i;
            } else {
                inf_count++;
            }
            continue;
        }
        if (!minmax_init) {
            min_v = max_v = v;
            minmax_init = true;
        } else {
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
    }
    if (nan_count > 0 || inf_count > 0) {
        fprintf(stderr, "[SPEC DEBUG] %s: logits non-finite (idx=%d) nan=%d inf=%d first_nan=%d min=%.6f max=%.6f\n",
                tag, token_index, nan_count, inf_count, first_nan, min_v, max_v);
        if (first_nan >= 0) {
            fprintf(stderr, "[SPEC DEBUG] %s: logits around first_nan:", tag);
            for (int i = std::max(0, first_nan - 4); i < std::min(n_vocab, first_nan + 5); ++i) {
                fprintf(stderr, " %.6f", logits[i]);
            }
            fprintf(stderr, "\n");
        }
    }
}

static bool get_moe_info(
    llama_context * ctx,
    const llama_model * model,
    int token_index,
    float * out_tae,
    ExpertLayers * out_experts,
    std::vector<float> * out_layer_ratios,
    std::vector<int> * out_layer_counts,
    float * out_ratio_avg,
    bool * warned_no_moe) {
    const int n_layers = llama_model_n_layer(model);
    std::vector<llama_moe_topk_layer> layers(n_layers);
    int n_out_layers = 0;
    static int moe_debug_left = -1;
    if (moe_debug_left < 0) {
        const char * env = std::getenv("SPEC_MOE_DEBUG");
        if (env && *env) {
            moe_debug_left = std::max(0, std::atoi(env));
        } else {
            moe_debug_left = 0;
        }
    }

    if (!llama_get_last_moe_topk(ctx, token_index, layers.data(), n_layers, &n_out_layers) || n_out_layers == 0) {
        if (warned_no_moe && !*warned_no_moe) {
            // LOG_INF("MoE routing info unavailable; using proxy TAE/cost\n");
            *warned_no_moe = true;
        }
        if (out_tae) {
            *out_tae = 0.5f;
        }
        if (out_experts) {
            out_experts->resize(n_layers);
            out_experts->clear();
        }
        if (out_layer_ratios) {
            out_layer_ratios->clear();
        }
        if (out_layer_counts) {
            out_layer_counts->clear();
        }
        if (out_ratio_avg) {
            *out_ratio_avg = -1.0f;
        }
        if (moe_debug_left > 0) {
            fprintf(stderr, "[SPEC MOE] token=%d no_moe n_out_layers=%d\n", token_index, n_out_layers);
            moe_debug_left--;
        }
        return false;
    }

    float tae_sum = 0.0f;
    int tae_count = 0;
    float ratio_sum = 0.0f;
    int ratio_count = 0;

    if (out_experts) {
        out_experts->resize(n_layers);
        out_experts->clear();
    }
    if (out_layer_ratios) {
        out_layer_ratios->assign(n_layers, -1.0f);
    }
    if (out_layer_counts) {
        out_layer_counts->assign(n_layers, -1);
    }

    if (moe_debug_left > 0) {
        fprintf(stderr, "[SPEC MOE] token=%d n_out_layers=%d expert_max=%d\n",
                token_index, n_out_layers, g_prune_config.expert_max);
    }

    for (int i = 0; i < n_out_layers; ++i) {
        const int n = std::min(layers[i].n, g_prune_config.m_route);
        if (n <= 0) {
            continue;
        }

        // Always record top-k experts for cost accounting, even if scores are all <= 0
        if (out_experts) {
            const int k = std::min(layers[i].n, g_prune_config.k_tgt);
            int invalid = 0;
            int min_id = INT_MAX;
            int max_id = INT_MIN;
            for (int j = 0; j < k; ++j) {
                const int layer_id = layers[i].layer;
                const int id = layers[i].expert_id[j];
                if (id < 0 || id >= g_prune_config.expert_max) {
                    invalid++;
                } else {
                    min_id = std::min(min_id, id);
                    max_id = std::max(max_id, id);
                    out_experts->set(layer_id, id);
                }
            }
            if (moe_debug_left > 0) {
                fprintf(stderr, "[SPEC MOE] layer=%d n=%d k=%d invalid=%d min_id=%d max_id=%d\n",
                        layers[i].layer, layers[i].n, k, invalid,
                        min_id == INT_MAX ? -1 : min_id,
                        max_id == INT_MIN ? -1 : max_id);
            }
        }

        float sum_w = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum_w += std::max(layers[i].score[j], 0.0f);
        }
        if (sum_w <= 0.0f) {
            // Skip TAE/ratio statistics if logits are non-positive, but keep experts
            continue;
        }

        float H = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float w = std::max(layers[i].score[j], 0.0f) / sum_w;
            H -= w * std::log(w + g_prune_config.eps);
        }
        const float denom = std::log(std::max(2.0f, (float) n));
        tae_sum += denom > 0.0f ? H / denom : 0.0f;
        tae_count += 1;

        if (out_layer_ratios) {
            const int layer_id = layers[i].layer;
            const int n_top = std::min(layers[i].n, LLAMA_MOE_TOPK_MAX);
            std::vector<float> scores(n_top);
            float max_score = -INFINITY;
            for (int j = 0; j < n_top; ++j) {
                scores[j] = layers[i].score[j];
                max_score = std::max(max_score, scores[j]);
            }
            float sum_exp = 0.0f;
            for (int j = 0; j < n_top; ++j) {
                scores[j] = std::exp(scores[j] - max_score);
                sum_exp += scores[j];
            }
            if (sum_exp > 0.0f) {
                for (int j = 0; j < n_top; ++j) {
                    scores[j] /= sum_exp;
                }
            }
            std::sort(scores.begin(), scores.end(), std::greater<float>());
            const int n8 = std::min(8, n_top);
            const int n3 = std::min(3, n_top);
            float sum8 = 0.0f;
            float sum3 = 0.0f;
            for (int j = 0; j < n8; ++j) {
                sum8 += scores[j];
            }
            for (int j = 0; j < n3; ++j) {
                sum3 += scores[j];
            }
            const float ratio = sum8 > 0.0f ? (sum3 / sum8) : 0.0f;
            if (layer_id >= 0 && layer_id < (int) out_layer_ratios->size()) {
                (*out_layer_ratios)[layer_id] = ratio;
            }
            if (out_layer_counts && layer_id >= 0 && layer_id < (int) out_layer_counts->size()) {
                (*out_layer_counts)[layer_id] = layers[i].n;
            }
            ratio_sum += ratio;
            ratio_count += 1;
        }
    }

    if (moe_debug_left > 0) {
        moe_debug_left--;
    }

    if (out_tae) {
        *out_tae = tae_count > 0 ? tae_sum / tae_count : 0.5f;
    }
    if (out_ratio_avg) {
        *out_ratio_avg = ratio_count > 0 ? ratio_sum / ratio_count : 0.0f;
    }

    return true;
}

static std::string json_escape(const std::string & input) {
    std::string out;
    out.reserve(input.size());
    for (char c : input) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    out += "\\u00";
                    const char hex[] = "0123456789abcdef";
                    out += hex[(c >> 4) & 0x0f];
                    out += hex[c & 0x0f];
                } else {
                    out += c;
                }
                break;
        }
    }
    return out;
}

static std::string format_json_float_array(const std::vector<float> & values) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss << std::setprecision(4);
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

static std::string format_json_int_array(const std::vector<int> & values) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

static constexpr float k_ratio_bin_min = 0.5f;
static constexpr float k_ratio_bin_max = 0.7f;
static constexpr int k_ratio_bins = 10;
static long long g_ratio_bin_total[k_ratio_bins] = {};
static long long g_ratio_bin_accept[k_ratio_bins] = {};
static long long g_ratio_missing = 0;
static bool g_moe_log_enabled = false;

static void update_ratio_stats(float avg_ratio, bool accepted) {
    if (avg_ratio < 0.0f) {
        g_ratio_missing++;
        return;
    }
    const float span = k_ratio_bin_max - k_ratio_bin_min;
    const float rel = (avg_ratio - k_ratio_bin_min) / span;
    int bin = static_cast<int>(rel * k_ratio_bins);
    if (bin >= k_ratio_bins) {
        bin = k_ratio_bins - 1;
    }
    g_ratio_bin_total[bin]++;
    if (accepted) {
        g_ratio_bin_accept[bin]++;
    }
}

static void log_draft_token_moe(
    const char * status,
    bool accepted,
    int seq,
    int pos,
    llama_token token_id,
    const std::string & token_str,
    const seq_draft & draft) {
    if (!g_moe_log_enabled) {
        return;
    }
    if (pos < 0 || (size_t) pos >= draft.moe_ratio_avg.size()) {
        update_ratio_stats(-1.0f, accepted);
        LOG_INF("\nMOE_LOG {\"status\":\"%s\",\"seq\":%d,\"pos\":%d,\"token\":%d,\"piece\":\"%s\",\"ratio_avg\":null}\n",
                status, seq, pos, token_id, json_escape(token_str).c_str());
        return;
    }
    const float avg_ratio = draft.moe_ratio_avg[pos];
    update_ratio_stats(avg_ratio, accepted);
    const std::vector<float> & layer_ratios = draft.moe_layer_ratios[pos];
    const std::vector<int> & layer_counts = draft.moe_layer_counts[pos];
    const std::string ratios_json = format_json_float_array(layer_ratios);
    const std::string counts_json = format_json_int_array(layer_counts);
    if (avg_ratio < 0.0f) {
        LOG_INF("\nMOE_LOG {\"status\":\"%s\",\"seq\":%d,\"pos\":%d,\"token\":%d,\"piece\":\"%s\",\"ratio_avg\":null}\n",
                status, seq, pos, token_id, json_escape(token_str).c_str());
        return;
    }
    LOG_INF("\nMOE_LOG {\"status\":\"%s\",\"seq\":%d,\"pos\":%d,\"token\":%d,\"piece\":\"%s\",\"ratio_avg\":%.4f,\"layer_ratios\":%s,\"layer_counts\":%s}\n",
            status, seq, pos, token_id, json_escape(token_str).c_str(), avg_ratio, ratios_json.c_str(), counts_json.c_str());
}

static void log_ratio_distribution() {
    if (!g_moe_log_enabled) {
        return;
    }
    LOG_INF("\nMoE ratio_avg distribution (top3/top8), range [%.1f, %.1f):\n", k_ratio_bin_min, k_ratio_bin_max);
    for (int i = 0; i < k_ratio_bins; ++i) {
        const float step = (k_ratio_bin_max - k_ratio_bin_min) / k_ratio_bins;
        const float lo = k_ratio_bin_min + step * i;
        const float hi = k_ratio_bin_min + step * (i + 1);
        const long long total = g_ratio_bin_total[i];
        const long long accept = g_ratio_bin_accept[i];
        const float rate = total > 0 ? (100.0f * accept / total) : 0.0f;
        LOG_INF("  [%.1f, %.1f): total=%lld accepted=%lld accept_rate=%.2f%%\n", lo, hi, total, accept, rate);
    }
    LOG_INF("  missing_ratio_avg=%lld\n", g_ratio_missing);
}

static void draft_tree_original(
    llama_context * ctx_dft,
    const llama_model * model_dft,
    llama_memory_t mem_dft,
    llama_batch & batch_dft,
    llama_batch & batch_tgt,
    std::vector<seq_draft> & drafts,
    int n_seq_dft,
    int n_draft,
    float p_draft_split,
    int n_past_tgt,
    int n_past_dft,
    int & n_drafted) {
    int n_seq_cur  = 1;
    int n_past_cur = n_past_dft;
    bool warned_no_moe = false;

    // sample n_draft tokens from the draft model using tree-based sampling
    for (int i = 0; i < n_draft; ++i) {
        batch_dft.n_tokens = 0;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].skip = false;
        }

        std::vector<int> seqs_in_batch;
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].drafting || drafts[s].skip) {
                continue;
            }

            common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);

            const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);

            for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            std::vector<int> sa(1, s);

            // attempt to split the branch if the probability is high enough
            for (int f = 1; f < 8; ++f) {
                if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                    LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur);

                    llama_memory_seq_rm(mem_dft,    n_seq_cur, -1, -1);
                    llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);

                    // all previous tokens from this branch are now also part of the new branch
                    for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                        for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                            if (batch_tgt.seq_id[t][p] == s) {
                                batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                batch_tgt.n_seq_id[t]++;
                                break;
                            }
                        }
                    }

                    // copy the draft state
                    drafts[n_seq_cur].active   = true;
                    drafts[n_seq_cur].drafting = true;
                    drafts[n_seq_cur].skip     = true;

                    drafts[n_seq_cur].tokens      = drafts[s].tokens;
                    // ================= [新增：复制概率历史] =================
                    drafts[n_seq_cur].probs       = drafts[s].probs;
                    // ========================================================
                    drafts[n_seq_cur].dists       = drafts[s].dists;
                    drafts[n_seq_cur].moe_layer_ratios = drafts[s].moe_layer_ratios;
                    drafts[n_seq_cur].moe_ratio_avg    = drafts[s].moe_ratio_avg;
                    drafts[n_seq_cur].moe_layer_counts = drafts[s].moe_layer_counts;
                    drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                    drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                    if (drafts[n_seq_cur].smpl) {
                        common_sampler_free(drafts[n_seq_cur].smpl);
                    }
                    drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);

                    sa.push_back(n_seq_cur);

                    n_seq_cur++;
                } else {
                    break;
                }
            }

            // add drafted token for each sequence
            for (int is = 0; is < (int) sa.size(); ++is) {
                const llama_token id = cur_p->data[is].id;
                // ================= [新增：获取当前 Token 概率] =================
                const float p_token  = cur_p->data[is].p; 
                // =============================================================
                const int s = sa[is];

                common_sampler_accept(drafts[s].smpl, id, true);

                drafts[s].tokens.push_back(id);
                // ================= [新增：存入概率] =================
                drafts[s].probs.push_back(p_token);
                // ===================================================

                // save cur_p.data into drafts[s].dists
                drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                // add unique drafted tokens to the target batch
                drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                // add the token to the batch for batched decoding with the draft model
                drafts[s].i_batch_dft = batch_dft.n_tokens;

                common_batch_add(batch_dft, id, n_past_cur, { s }, true);
                seqs_in_batch.push_back(s);

                // if (batch_tgt.n_tokens > n_draft) {
                //     drafts[s].drafting = false;
                // }
            }
        }

        // no sequence is drafting anymore
        if (batch_dft.n_tokens == 0) {
            break;
        }

        // evaluate the drafted tokens on the draft model
        {
            std::string range_name ="draft_decode_n" + std::to_string(n_drafted);
            NvtxRange r_draft(range_name.c_str());
            llama_decode(ctx_dft, batch_dft);
        }
        
        ++n_past_cur;
        ++n_drafted;

        for (int s : seqs_in_batch) {
            std::vector<float> layer_ratios;
            std::vector<int> layer_counts;
            float avg_ratio = -1.0f;
            get_moe_info(ctx_dft, model_dft, drafts[s].i_batch_dft, nullptr, nullptr, &layer_ratios, &layer_counts, &avg_ratio, &warned_no_moe);
            draft_moe_push(drafts[s], layer_ratios, layer_counts, avg_ratio);
        }

        // if (batch_tgt.n_tokens > n_draft) {
        //     break;
        // }
    }
}

static void draft_tree_prune(
    llama_context * ctx_dft,
    const llama_model * model_dft,
    llama_memory_t mem_dft,
    llama_batch & batch_dft,
    llama_batch & batch_tgt,
    std::vector<seq_draft> & drafts,
    int n_seq_dft,
    int n_draft,
    int n_past_tgt,
    int n_past_dft,
    int max_depth,
    int max_nodes,
    int & n_drafted,
    StepTrace & current_trace) {
    max_depth = std::max(1, max_depth);
    max_nodes = std::max(1, max_nodes);
    // max_nodes = std::max(max_nodes, n_draft);

    int n_seq_cur  = 1;
    int n_past_cur = n_past_dft;

    struct Candidate {
        int seq = -1;
        int parent_seq = -1;
        bool new_seq = false;
        int i_batch_dft = -1;
        int depth = 0;
        llama_token id = LLAMA_TOKEN_NULL;
        float c_t = 0.0f;
        float tae = 0.5f;
        float alpha = 0.0f;
        float a = 1.0f;
        float H = 0.0f;
        bool has_moe = false;
        ExpertLayers token_experts;
        ExpertLayers predicted_experts;
        ExpertLayers union_experts;
        ExpertLayers new_experts;
        std::vector<float> moe_ratio_layers;
        std::vector<int> moe_ratio_layer_counts;
        float moe_ratio_avg = -1.0f;
        float score = 0.0f;
        float delta_c = 0.0f;
    };

    int selected_nodes = 0;
    float budget_used = 0.0f;
    int budget_used_experts = 0;
    const int n_layers = llama_model_n_layer(model_dft);
    ExpertLayers u_cache(n_layers);
    u_cache.clear();

    std::vector<float> seq_a_selected(n_seq_dft, 1.0f);
    std::vector<float> seq_H_selected(n_seq_dft, 0.0f);
    std::vector<int> seq_depth_selected(n_seq_dft, 0);
    std::vector<ExpertLayers> seq_u_selected(n_seq_dft, ExpertLayers(n_layers));
    std::vector<bool> active_seq(n_seq_dft, false);
    active_seq[0] = true;

    bool warned_no_moe = false;
    // 清空 trace 节点
    current_trace.nodes.clear();
    current_trace.total_cost = 0;
    for (int step = 0; step < n_draft && step < max_depth; ++step) {
        batch_dft.n_tokens = 0;
        float cur=1.0f;
        common_sampler_sample(drafts[0].smpl, ctx_dft, drafts[0].i_batch_dft, true);
        const auto * cur_p_temp = common_sampler_get_candidates(drafts[0].smpl, true);
        cur=cur_p_temp->data[0].p;
        if (!std::isfinite(cur)) {
            debug_dump_logits_if_nan(ctx_dft, model_dft, drafts[0].i_batch_dft, "draft_prune_head");
        }
        // only collect very high-confidence draft tokens
        // if (cur < 0.45) {
        //     // printf("only collect very high-confidence draft tokens!Now confidence is %.4f\n",cur);
        //     break;
        // }
        const std::vector<bool> active_before = active_seq;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].skip = false;
        }

        std::vector<Candidate> candidates;

        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].drafting || drafts[s].skip) {
                continue;
            }

            common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
            const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);
            if (!std::isfinite(cur_p->data[0].p)) {
                debug_dump_logits_if_nan(ctx_dft, model_dft, drafts[s].i_batch_dft, "draft_prune_cand");
            }
            // =========================================================================
            // [新增] 手动计算用于剪枝的置信度 (Confidence)，绕过 sampler 的内部逻辑
            // =========================================================================
            
            // 1. 定义剪枝专用的温度 (可以从 g_prune_config 读，这里先硬编码测试)
            //    建议设为 1.0 到 2.0 之间。如果设为 10.0，概率会极度平滑。
            float prune_temp = 1.0f; // 你之前设 10.0 太大了，先试试 1.0 标准值

            // 2. 找到最大 Logit (用于数值稳定，防止 exp 溢出)
            // float max_logit = -1e9f;
            // for (size_t i = 0; i < cur_p->size; ++i) {
            //     if (cur_p->data[i].logit > max_logit) {
            //         max_logit = cur_p->data[i].logit;
            //     }
            // }

            // // 3. 计算分母 (Sum of Exps)
            // double sum_exp = 0.0;
            // for (size_t i = 0; i < cur_p->size; ++i) {
            //     // 如果 prune_temp 很大，这里差异变小；如果很小，差异变大
            //     sum_exp += std::exp((cur_p->data[i].logit - max_logit) / prune_temp);
            // }

            // // [调试打印] 看看前几个 Logit 是多少，验证是否巨大
            // if (step == 0 && s == 0) {
            //      LOG_INF("DEBUG: max_logit=%f, sum_exp=%f\n", max_logit, sum_exp);
            //      // 打印前3个候选的 logit 和 手动算的 p
            //      for(int k=0; k<3 && k<cur_p->size; ++k) {
            //          float l = cur_p->data[k].logit;
            //          float p_manual = std::exp((l - max_logit) / prune_temp) / sum_exp;
            //          LOG_INF("  Top-%d: id=%d, logit=%f, p_orig=%f, p_manual=%f\n", 
            //                  k, cur_p->data[k].id, l, cur_p->data[k].p, p_manual);
            //      }
            // }
            // =========================================================================
            const int available = std::max(0, n_seq_dft - n_seq_cur);
            int n_candidates = std::min((int) cur_p->size, 1 + available);
            if (step == 0) {
                n_candidates = std::min(n_candidates, 1);
            }


            if (n_candidates <= 0) {
                continue;
            }

            std::vector<int> sa(1, s);
            for (int f = 1; f < n_candidates; ++f) {
                float val = cur_p->data[f].logit;
                // float p_manual = std::exp((val - max_logit) / prune_temp) / sum_exp;
                if (n_seq_cur < n_seq_dft && cur_p->data[f].p > 0.1) { // step==0 延迟扩分支
                    llama_memory_seq_rm(mem_dft,    n_seq_cur, -1, -1);
                    llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);

                    drafts[n_seq_cur].active   = true;
                    drafts[n_seq_cur].drafting = true;
                    drafts[n_seq_cur].skip     = true;

                    drafts[n_seq_cur].tokens      = drafts[s].tokens;
                    // ================= [新增：复制概率历史] =================
                    drafts[n_seq_cur].probs       = drafts[s].probs;
                    // ========================================================
                    drafts[n_seq_cur].dists       = drafts[s].dists;
                    drafts[n_seq_cur].moe_layer_ratios = drafts[s].moe_layer_ratios;
                    drafts[n_seq_cur].moe_ratio_avg    = drafts[s].moe_ratio_avg;
                    drafts[n_seq_cur].moe_layer_counts = drafts[s].moe_layer_counts;
                    drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                    drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                    if (drafts[n_seq_cur].smpl) {
                        common_sampler_free(drafts[n_seq_cur].smpl);
                    }
                    drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);

                    sa.push_back(n_seq_cur);
                    n_seq_cur++;
                }
            }

            for (int is = 0; is < (int) sa.size(); ++is) {
                const int seq_id = sa[is];
                const llama_token id = cur_p->data[is].id;
                // ================= [新增：获取当前 Token 概率] =================
                const float p_token  = cur_p->data[is].p; 
                // =============================================================
                drafts[seq_id].tokens.push_back(id);
                // ================= [新增：存入概率] =================
                drafts[seq_id].probs.push_back(p_token);
                // ===================================================
                drafts[seq_id].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                drafts[seq_id].i_batch_dft = batch_dft.n_tokens;
                common_batch_add(batch_dft, id, n_past_cur, { seq_id }, true);

                Candidate cand;
                cand.seq = seq_id;
                cand.parent_seq = s;
                cand.new_seq = seq_id != s;
                cand.i_batch_dft = drafts[seq_id].i_batch_dft;
                cand.depth = step + 1;
                cand.id = id;
                cand.c_t = cur_p->data[is].p;
                cand.token_experts.resize(n_layers);
                cand.predicted_experts.resize(n_layers);
                cand.union_experts.resize(n_layers);
                cand.new_experts.resize(n_layers);
                candidates.push_back(std::move(cand));
                // [关键修改] 使用手动计算的概率赋值给 c_t
                float val = cur_p->data[is].logit;
                // float p_manual = std::exp((val - max_logit) / prune_temp) / sum_exp;
                
                // cand.c_t = p_manual; // <--- 这里！不要用 cur_p->data[is].p
            }
            

        }

        if (batch_dft.n_tokens == 0) {
            break;
        }
        

        float delta_g;
        float delta_c;

        for (auto & cand : candidates) {
            cand.has_moe = get_moe_info(ctx_dft, model_dft, cand.i_batch_dft, &cand.tae, &cand.token_experts, &cand.moe_ratio_layers, &cand.moe_ratio_layer_counts, &cand.moe_ratio_avg, &warned_no_moe);
            const uint64_t noise_seed =
                static_cast<uint64_t>(g_prune_config.routing_noise_seed) ^
                (static_cast<uint64_t>(current_trace.step_idx + 1) << 32) ^
                (static_cast<uint64_t>(cand.depth + 1) << 24) ^
                (static_cast<uint64_t>(cand.seq + 1) << 16) ^
                static_cast<uint64_t>(static_cast<uint32_t>(cand.id));
            cand.predicted_experts = perturb_predicted_experts(cand.token_experts, noise_seed);
            cand.alpha = compute_alpha(cand.c_t, cand.tae);
            cand.a = seq_a_selected[cand.parent_seq] * cand.alpha;
            cand.H = seq_H_selected[cand.parent_seq] + cand.a;
            draft_moe_push(drafts[cand.seq], cand.moe_ratio_layers, cand.moe_ratio_layer_counts, cand.moe_ratio_avg);

            cand.union_experts = seq_u_selected[cand.parent_seq];
            if (cand.has_moe) {
                cand.union_experts.or_with(cand.predicted_experts);
            }

            // const float delta_g = (cand.H - seq_H_selected[cand.parent_seq]) * PRUNE_TPOT;
            // delta_g = (cand.a*(step+1) - seq_a_selected[cand.parent_seq]*step) * PRUNE_TPOT;
            delta_g = (cand.a) * g_prune_config.tpot;

            delta_c = 0.0f;
            if (cand.has_moe) {
                ExpertLayers u_seg = cand.union_experts;
                u_seg.andnot_with(seq_u_selected[cand.parent_seq]);
                cand.new_experts = u_seg;
                cand.new_experts.andnot_with(u_cache);
                delta_c = cand.new_experts.popcount() * (g_prune_config.expert_bytes / g_prune_config.bandwidth);
                // printf("activate %d experts\n", cand.new_experts.popcount());
            } else {
                delta_c = g_prune_config.expert_bytes / g_prune_config.bandwidth;
            }

            // Ensure we start counting from step 0 even if no experts are reported
            if (step == 0 && delta_c == 0.0f) {
                delta_c = g_prune_config.expert_bytes / g_prune_config.bandwidth;
            }

            cand.delta_c = delta_c;
            cand.score = delta_g / (delta_c + g_prune_config.eps);
        }

        std::sort(candidates.begin(), candidates.end(), [](const Candidate & a, const Candidate & b) {
            return a.score > b.score;
        });

        std::vector<bool> selected_seq(n_seq_dft, false);
        float score_sum = 0.0f;
        float score_max = 0.0f;
        int score_count = 0;

        for (const auto & cand : candidates) {
            if (selected_nodes >= max_nodes) {
                break;
            }
            if (cand.depth > max_depth) {
                continue;
            }
            if (budget_used + cand.delta_c > g_prune_config.budget_b) {
                continue;
            }
            if (step == 0) {
                if (cand.c_t < 0.5f) {
                    continue;
                }
            } else {
                if (cand.score < 0.8) {
                    continue;
                }
            }
            if (batch_tgt.n_tokens >= n_draft) {
                break;
            }
            // [新增] 记录 Trace Node
            TraceNode node;
            node.node_id = cand.seq; // 这里的 seq 是 draft tree 的唯一标识
            node.parent_id = cand.parent_seq;
            node.token_id = cand.id;
            node.token_str = common_token_to_piece(ctx_dft, cand.id);
            node.depth = cand.depth;
            node.confidence = cand.c_t;
            node.cost = cand.delta_c;
            node.accepted = false; // 初始为 false，后面验证时更新
            int target_idx = batch_tgt.n_tokens; // [新增]
            node.tmp_target_batch_idx = target_idx;
            
            // 收集 Draft Experts
            // 注意：capture_experts 需要 batch 中的 index
            // 这里 cand.i_batch_dft 就是 draft batch 中的索引
            // 但是 draft model 此时可能还没有 decode (如果 decode 在下面)
            // 检查代码发现：llama_decode(ctx_dft, batch_dft) 在循环中间调用了
            // 所以此时可以直接获取 expert info
            if (g_prune_config.enable_trace) {
                node.draft_experts = capture_experts(ctx_dft, cand.i_batch_dft);
                node.predicted_experts = expert_layers_to_activations(cand.predicted_experts);
                
                // 收集 Draft Logits
                node.draft_logits_topk = get_logits_topk(ctx_dft, cand.i_batch_dft, 5);
            }
            current_trace.nodes.push_back(node);
            const int selected_new_experts = cand.has_moe ? cand.new_experts.popcount() : 1;
            current_trace.total_cost += cand.delta_c;
            budget_used += cand.delta_c;
            budget_used_experts += selected_new_experts;
            selected_nodes++;
            selected_seq[cand.seq] = true;

            if (cand.has_moe) {
                u_cache.or_with(cand.new_experts);
            }

            seq_a_selected[cand.seq] = cand.a;
            seq_H_selected[cand.seq] = cand.H;
            seq_depth_selected[cand.seq] = cand.depth;
            seq_u_selected[cand.seq] = cand.union_experts;

            if (cand.new_seq) {
                for (int t = 0; t < (int) drafts[cand.seq].i_batch_tgt.size(); ++t) {
                    const int idx = drafts[cand.seq].i_batch_tgt[t];
                    bool exists = false;
                    for (int p = 0; p < batch_tgt.n_seq_id[idx]; ++p) {
                        if (batch_tgt.seq_id[idx][p] == cand.seq) {
                            exists = true;
                            break;
                        }
                    }
                    if (!exists) {
                        batch_tgt.seq_id[idx][batch_tgt.n_seq_id[idx]] = cand.seq;
                        batch_tgt.n_seq_id[idx]++;
                    }
                }
            }
            // LOG_INF("prune step %d: |S|=%d, C=%.4f/%.4f, delta_g=%.4f,c_t=%.4f,alpha=%.4f,a=%.4f,delta_c=%.4f\n",
            //         step, selected_nodes, budget_used, g_prune_config.budget_b, delta_g ,cand.c_t,cand.alpha,cand.a, (delta_c + g_prune_config.eps));
            common_sampler_accept(drafts[cand.seq].smpl, cand.id, true);

            drafts[cand.seq].i_batch_tgt.push_back(batch_tgt.n_tokens);

            common_batch_add(batch_tgt, cand.id, n_past_tgt + step + 1, { cand.seq }, true);

            score_sum += cand.score;
            score_max = std::max(score_max, cand.score);
            score_count++;
        }

        for (const auto & cand : candidates) {
            if (selected_seq[cand.seq] || !active_before[cand.seq]) {
                continue;
            }
            if (!drafts[cand.seq].tokens.empty()) {
                drafts[cand.seq].tokens.pop_back();
            }
            if (!drafts[cand.seq].dists.empty()) {
                drafts[cand.seq].dists.pop_back();
            }
            if (!drafts[cand.seq].moe_layer_ratios.empty()) {
                drafts[cand.seq].moe_layer_ratios.pop_back();
            }
            if (!drafts[cand.seq].moe_ratio_avg.empty()) {
                drafts[cand.seq].moe_ratio_avg.pop_back();
            }
            if (!drafts[cand.seq].moe_layer_counts.empty()) {
                drafts[cand.seq].moe_layer_counts.pop_back();
            }
            llama_memory_seq_rm(mem_dft, cand.seq, n_past_dft + step, -1);
        }

        for (int s = 0; s < n_seq_dft; ++s) {
            active_seq[s] = active_seq[s] || selected_seq[s];
            drafts[s].active = active_seq[s];
            drafts[s].drafting = selected_seq[s];
        }
        if (false) {
            if (score_count > 0) {
                LOG_INF("prune step %d: |S|=%d, C=%.4f/%.4f, E=%d, score avg=%.4f max=%.4f delta_g=%.4f,delta_c=%.4f\n",
                        step, selected_nodes, budget_used, g_prune_config.budget_b, budget_used_experts,
                        score_sum / score_count, score_max,delta_g ,(delta_c + g_prune_config.eps));
            } else {
                LOG_INF("prune step %d: |S|=%d, C=%.4f/%.4f, E=%d\n",
                        step, selected_nodes, budget_used, g_prune_config.budget_b, budget_used_experts);
            }
        }
        if (selected_nodes >= max_nodes || batch_tgt.n_tokens >= n_draft) {
            break;
        }

        bool any_active = false;
        for (int s = 0; s < n_seq_dft; ++s) {
            if (drafts[s].drafting) {
                any_active = true;
                break;
            }
        }
        if (!any_active) {
            break;
        }
        {
            std::string range_name ="draft_decode_n" + std::to_string(n_drafted);
            NvtxRange r_draft(range_name.c_str());
            llama_decode(ctx_dft, batch_dft);
        }

        // llama_decode(ctx_dft, batch_dft);
        ++n_past_cur;
        ++n_drafted;
    }
}

int main(int argc, char ** argv) {
    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }
    if (const char * layer_profile_path = getenv("SPEC_LAYER_PROFILE_JSON")) {
        if (layer_profile_path[0] != '\0') {
            g_layer_profile.configured = true;
            g_layer_profile.filename = layer_profile_path;
            params.cb_eval = layer_profile_cb;
            params.cb_eval_user_data = nullptr;
            LOG_INF("%s: target verification layer profile enabled: %s\n",
                    __func__, g_layer_profile.filename.c_str());
        }
    }
    // 3. 将原代码中使用 PRUNE_BUDGET_B 的地方替换为 params.sprune.budget_b
    // 示例：
    // float current_budget = PRUNE_BUDGET_B; 
    // 改为 ->
    float current_budget = params.sprune.budget_b;
    
    // 如果你的剪枝逻辑在单独的函数里，记得把 params.sprune 传进去
    // 或者为了改动最小，可以在 main 函数开头定义局部变量：
    // =================================================================
    // [新增] 将命令行参数同步到全局配置对象中
    // =================================================================
    g_prune_config.budget_b      = params.sprune.budget_b;
    g_prune_config.m_route       = params.sprune.m_route;
    g_prune_config.k_tgt         = params.sprune.k_tgt;
    g_prune_config.beta          = params.sprune.beta;
    g_prune_config.gamma         = params.sprune.gamma;
    g_prune_config.lambda        = params.sprune.lambda;
    g_prune_config.tpot          = params.sprune.tpot;
    g_prune_config.eps           = params.sprune.eps;
    g_prune_config.expert_bytes  = params.sprune.expert_bytes;
    g_prune_config.bandwidth     = params.sprune.bandwidth;
    g_prune_config.expert_max    = params.sprune.expert_max;
    g_prune_config.routing_noise = params.sprune.routing_noise;
    g_prune_config.routing_noise_seed = params.sprune.routing_noise_seed;
    g_prune_config.routing_noise_experts = params.sprune.routing_noise_experts;
    g_capture_expert_count = params.sprune.routing_noise_experts;
    g_prune_config.enable_trace  = params.sprune.enable_trace;
    if (g_prune_config.enable_trace) {
        LOG_INF("Detailed trace collection ENABLED. Performance may decrease.\n");
    }
    if (params.speculative.moe_utility_spec) {
        LOG_INF("%s: MoE utility-driven speculation (Cascade-style): --draft cap=%d test_iters=%d set_iters=%d\n",
                __func__, params.speculative.n_max, params.speculative.utility_test_iters, params.speculative.utility_set_iters);
    }
    if (g_prune_config.routing_noise > 0.0f) {
        LOG_INF("%s: routing-noise sensitivity rate=%.3f seed=%d experts/layer=%d\n",
                __func__,
                g_prune_config.routing_noise,
                g_prune_config.routing_noise_seed,
                g_prune_config.routing_noise_experts);
    }
    // =================================================================
    common_init();
    // params.sampling.top_k = 0;
    // params.sampling.top_p = 1.0f;
    // params.sampling.min_p = 0.0f;    // 禁用 Min-P (默认通常是 0.05)
    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    const float p_draft_split = params.speculative.p_split;

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    if (params.speculative.prune == 1 && params.n_parallel > 1 && !params.kv_unified) {
        LOG_INF("%s: enabling KV unified mode for prune with parallel sequences\n", __func__);
        params.kv_unified = true;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();
    // =============== [新增] 初始化 Chat Template ===============
    // 自动侦测模型的模版格式 (Llama-3, ChatML, ChatGLM 等)
    auto chat_templates = common_chat_templates_init(model_tgt, params.chat_template);
    bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());
    
    if (has_chat_template) {
        LOG_INF("%s: chat template detected and enabled.\n", __func__);
        if (!params.system_prompt.empty()) {
             LOG_INF("System prompt: %s\n", params.system_prompt.c_str());
        }
    } else {
        LOG_INF("%s: no chat template detected, using raw string concatenation.\n", __func__);
    }
    // =========================================================
    // load the draft model

    // ================= [修改] Self-Speculation + SSD Offload 逻辑 =================
    
    // 1. 判断是否是 Self-Speculation (没有指定 draft model 路径，或者路径相同)
    bool is_self_speculation = params.speculative.model.path.empty() || 
                               (params.speculative.model.path == params.model.path);

    // 用来管理 Draft 资源的容器 (如果是独立加载的话)
    common_init_result llama_init_dft; 

    if (is_self_speculation) {
        LOG_INF("%s: Enabling Self-Speculation (Reusing Target Model)\n", __func__);
        
        // [关键] 直接复用 Target 模型的指针！
        // 这样 Draft 和 Target 共享同一个 SSD Registry 和 Host Memory Cache
        model_dft = model_tgt; 

        // 为 Draft 创建独立的 Context
        // Draft 的 Context 需要较小的 Batch Size (n_draft + 1)
        common_params params_dft = params;
        params_dft.n_batch = 1024*(params.speculative.n_max + 1); 
        
        // 将 common_params 转换为 llama_context_params
        llama_context_params ctx_params_dft = common_context_params_to_llama(params_dft);
        
        // 手动创建 Context
        ctx_dft = llama_new_context_with_model(model_dft, ctx_params_dft);
        if (!ctx_dft) {
            LOG_ERR("%s: failed to create draft context\n", __func__);
            return 1;
        }

    } else {
        // [常规逻辑] 加载独立的 Draft 小模型
        LOG_INF("%s: Loading separate Draft Model...\n", __func__);
        params.devices = params.speculative.devices;
        params.model = params.speculative.model;
        params.n_gpu_layers = params.speculative.n_gpu_layers;
        if (params.speculative.cpuparams.n_threads > 0) {
            params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
        }

        
        params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
        params.tensor_buft_overrides     = params.speculative.tensor_buft_overrides;

        // [重要] Draft 小模型通常不需要 SSD Offload，强制关闭以防误伤
        bool old_ssd_setting = params.use_ssd_moe;
        params.use_ssd_moe = false; 

        llama_init_dft = common_init_from_params(params);
        
        params.use_ssd_moe = old_ssd_setting; // 恢复设置

        model_dft = llama_init_dft.model.get();
        ctx_dft   = llama_init_dft.context.get();
    }
    if (params.speculative.dflash) {
        llama_model_dflash_set_target_model(model_dft, model_tgt);
    }
    // common_init_result llama_init_dft = common_init_from_params(params);

    // model_dft = llama_init_dft.model.get();
    // ctx_dft   = llama_init_dft.context.get();
    // ================= [修改] 配置 Draft 模型的 MoE 行为 =================
    
    // 设置 Draft 激活的专家数量 (TopK)
    // 命令行参数: --draft-expert-topk 2 (例如 Target 是 4，Draft 设为 2)
    if (params.speculative.draft_expert_topk != -1) {
        LOG_INF("%s: Setting Draft Expert TopK = %d\n", __func__, params.speculative.draft_expert_topk);
        
        // 这个 API 会告诉后端在 ctx_dft 推理时只计算 Top K 个专家
        // 对于我们的 SSD Offload 逻辑，这意味着它只会调用 llama_ssd_get_expert 2次，而不是 4次
        // 直接减少 50% 的 IO 开销！
        llama_context_set_draft_expert_topk(ctx_dft, params.speculative.draft_expert_topk);
    }
    const int32_t trace_moe_log_k = g_prune_config.enable_trace
        ? std::min<int32_t>(LLAMA_MOE_TOPK_MAX, std::max<int32_t>(1, g_prune_config.k_tgt))
        : 0;
    llama_set_moe_topk(ctx_tgt, g_prune_config.enable_trace);
    llama_context_set_moe_topk_log_k(ctx_tgt, g_prune_config.enable_trace ? trace_moe_log_k : -1);
    llama_set_moe_topk(ctx_dft, true);
    llama_context_set_moe_topk_log_k(ctx_dft, trace_moe_log_k);
    llama_context_set_draft_context(ctx_dft, true);
    if (!params.speculative.draft_skip_attn_layers.empty() || !params.speculative.draft_skip_mlp_layers.empty()) {
        llama_context_set_draft_skip_layers(
            ctx_dft,
            params.speculative.draft_skip_attn_layers.data(),
            params.speculative.draft_skip_attn_layers.size(),
            params.speculative.draft_skip_mlp_layers.data(),
            params.speculative.draft_skip_mlp_layers.size());
    }
    if (params.speculative.draft_expert_topk != -1) {
        llama_context_set_draft_expert_topk(ctx_dft, params.speculative.draft_expert_topk);
    }
    if (!params.speculative.draft_layer_expert_topk.empty()) {
        llama_context_set_draft_layer_expert_topk(
            ctx_dft,
            params.speculative.draft_layer_expert_topk.data(),
            params.speculative.draft_layer_expert_topk.size());
    }
    // =============== 在这里插入修改 ===============

    // 检查是否适合共享 KV Cache (Self-Speculation)
    // 条件：Draft 和 Target 架构一致 (层数、维度等)
    if (llama_n_layer(model_tgt) == llama_n_layer(model_dft) &&
        llama_n_embd(model_tgt) == llama_n_embd(model_dft) &&
        llama_n_head(model_tgt) == llama_n_head(model_dft) && params.speculative.share_kv) {
        
        LOG_INF("%s: Enabling Self-Speculation Optimization (Shared KV Cache)\n", __func__);
        llama_share_kv_cache(ctx_dft, ctx_tgt);
        
    } else {
        LOG_INF("%s: Models are different, using independent KV Caches.\n", __func__);
    }

    // ===========================================
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);
    llama_ssd_profile_reset();
    const std::set<std::pair<int, int>> resident_experts = load_hot_expert_set(params.hot_experts_path);
    std::set<std::pair<int, int>> seen_target_experts;
    RoutingReuseStats routing_stats;
    LOG_INF("%s: loaded %zu resident expert entries from hot-experts metadata\n",
            __func__, resident_experts.size());
    g_moe_log_enabled = params.moe_log;
    const std::vector<int32_t> default_skip_attn = params.speculative.draft_skip_attn_layers;
    const std::vector<int32_t> default_skip_mlp = params.speculative.draft_skip_mlp_layers;
    const int32_t default_expert_topk = params.speculative.draft_expert_topk;
    const std::vector<int32_t> default_layer_topk = params.speculative.draft_layer_expert_topk;
    // ================= [修改] 模式判断逻辑 =================
    const bool use_search_stdin = params.speculative.search_stdin;
    bool use_dataset = !params.dataset_path.empty() || use_search_stdin;
    std::ifstream dataset_file;
    // [修改] 使用 params.dataset_path
    // 如果用户没传参数，可以给个默认值，或者报错
    // if (params.dataset_path.empty()) {
    //     // 你可以选择设置默认值
    //     params.dataset_path = "datasets/gsm8K.jsonl";
    //     LOG_INF("No dataset specified, defaulting to: %s\n", params.dataset_path.c_str());
    //     // 或者报错退出
    //     // LOG_ERR("--dataset argument is required\n");
    //     // return 1;
    // }
    // // [新增] 打开数据集文件
    // // std::ifstream dataset_file("datastes/OlymMATH-EN-EASY.jsonl");
    // LOG_INF("Loading dataset from: %s\n", params.dataset_path.c_str());
    // std::ifstream dataset_file(params.dataset_path);
    // if (!dataset_file.is_open()) {
    //     LOG_ERR("Failed to open dataset file\n");
    //     return 1;
    // }
    if (use_dataset && !use_search_stdin) {
        // [模式 A]: 数据集模式
        LOG_INF("Loading dataset from: %s\n", params.dataset_path.c_str());
        dataset_file.open(params.dataset_path);
        if (!dataset_file.is_open()) {
            LOG_ERR("Failed to open dataset file: %s\n", params.dataset_path.c_str());
            return 1;
        }
    } else {
        // [模式 B]: 单 Prompt 模式 (-p / --prompt)
        LOG_INF("No dataset specified. Running in single-prompt mode using '-p'.\n");
        if (params.prompt.empty()) {
            // 给个默认值防止空转
            params.prompt = "Hello, how are you?";
        }
    }
    std::string json_line;
    int question_idx = 0;
    // int n_predict = 0;
    // int n_drafted = 0;
    // int n_accept  = 0;
    // ================= [新增] 全局统计累加器 =================
    double total_t_enc_us = 0.0; // 总编码时间 (微秒)
    double total_t_dec_us = 0.0; // 总解码时间 (微秒)
    double total_t_verify_us = 0.0; // 总 target verification 时间 (微秒)
    
    long long total_n_input   = 0; // 总输入 token 数
    long long total_n_predict = 0; // 总生成 token 数
    long long total_n_drafted = 0; // 总草稿 token 数
    long long total_n_accept  = 0; // 总接受 token 数
    int processed_count       = 0; // 处理的问题总数
    // =======================================================
    // [新增] 开始循环处理每一行
    
    while (true) {
        std::string current_prompt;
        // 1. 获取输入 (Fetch Input)
        if (use_dataset) {
            // --- 数据集模式取数据 ---
            if (use_search_stdin) {
                if (!std::getline(std::cin, json_line)) {
                    break;
                }
            } else {
                if (!std::getline(dataset_file, json_line)) {
                    break; // 文件读完了，退出循环
                }
            }
            if (json_line.empty()) continue;

            // 检查数量限制 (如果有 --n-questions 参数)
            if (params.n_questions_limit > 0 && processed_count >= params.n_questions_limit) {
                LOG_INF("Reached question limit (%d). Stopping.\n", params.n_questions_limit);
                break;
            }

            question_idx++;
            // ================= [修改] 兼容不同数据集格式 =================
            std::string question;
            std::string dataset_type = "unknown";

            // 1. 尝试 GSM8K 格式 ("question": "...")
            if (use_search_stdin) {
                question = extract_json_value(json_line, "prompt");
                if (!question.empty()) {
                    dataset_type = "Prompt";
                }
            }
            if (question.empty()) {
                question = extract_json_value(json_line, "question");
                if (!question.empty()) {
                    dataset_type = "GSM8K";
                } else {
                    question = extract_json_value(json_line, "turns");
                    if (!question.empty()) {
                        dataset_type = "HumanEval";
                    } else {
                        question = extract_json_value(json_line, "prompt");
                        if (!question.empty()) {
                            dataset_type = "Generic";
                        }
                    }
                }
            }

            if (question.empty()) {
                LOG_WRN("Skipping line %d: could not find 'question', 'turns' or 'prompt' key.\n", question_idx);
                continue;
            }
            // std::string question = extract_json_value(json_line, "question");
            // if (question.empty()) continue;

            // 构造 Dataset 模式的 Prompt (GSM8K 风格)
            // current_prompt = "Question: " + question + "\nAnswer:";
            // current_prompt = "<|im_start|>user: "+question;
// ================= [修改] 支持 Chat Template =================
            if (has_chat_template) {
                // 1. 构建消息列表
                std::vector<common_chat_msg> messages;

                // (可选) 添加 System Prompt
                // 如果命令行没传 -sys，可以给个默认的，或者为空
                if (!params.system_prompt.empty()) {
                    // messages.push_back({"system", params.system_prompt});
                    messages.push_back({
                        "system",
                        "You are a helpful assistant. "
                        "Do NOT output analysis, reasoning, or chain-of-thought. "
                        "Only output the final answer."
                    });
                } 
                else {
                    // messages.push_back({"system", "You are a helpful assistant. Solve the math problem step by step."});
                    messages.push_back({
                        "system",
                        "You are a helpful assistant. "
                        "Do NOT output analysis, reasoning, or chain-of-thought. "
                        "Only output the final answer."
                    });
                
                }

                // 添加用户问题
                messages.push_back({"user", question});

                // 2. 应用模版
                common_chat_templates_inputs inputs;
                inputs.messages = messages;
                inputs.use_jinja = params.use_jinja;
                inputs.parallel_tool_calls = false;
                
                // [关键] 设置为 true，让模板生成 "assistant" 的起始 token (例如 <|im_start|>assistant\n)
                // 这样模型就知道该轮到它说话了
                inputs.add_generation_prompt = true; 

                // 生成最终 prompt
                current_prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
                
                // 调试打印：看看生成的 prompt 长什么样
                // LOG_INF("Formatted Chat Prompt:\n%s\n", current_prompt.c_str());

            } else {
                // [回退] 如果没有模板 (Base Model)，使用旧的拼接方式
                current_prompt = "<|im_start|>user\n"+question+"<|im_end|>\n";
            }
            // ============================================================
            LOG_INF("\n\n=== Processing Question %d (Dataset) ===\n", question_idx);

        } else {
            // --- 单 Prompt 模式取数据 ---
            if (processed_count >= 1) {
                break; // 已经跑完一次了，退出循环
            }
            
            // 直接使用命令行传入的 prompt
            if (has_chat_template) {
                // 1. 构建消息列表
                std::vector<common_chat_msg> messages;

                // (可选) 添加 System Prompt
                // 如果命令行没传 -sys，可以给个默认的，或者为空
                if (!params.system_prompt.empty()) {
                    // messages.push_back({"system", params.system_prompt});
                    messages.push_back({
                        "system",
                        "You are a helpful assistant. "
                        "Do NOT output analysis, reasoning, or chain-of-thought. "
                        "Only output the final answer."
                    });
                } 
                else {
                    // messages.push_back({"system", "You are a helpful assistant. Solve the math problem step by step."});
                    messages.push_back({
                        "system",
                        "You are a helpful assistant. "
                        "Do NOT output analysis, reasoning, or chain-of-thought. "
                        "Only output the final answer."
                    });
                }

                // 添加用户问题
                messages.push_back({"user", params.prompt});

                // 2. 应用模版
                common_chat_templates_inputs inputs;
                inputs.messages = messages;
                inputs.use_jinja = params.use_jinja;
                inputs.parallel_tool_calls = false;
                
                // [关键] 设置为 true，让模板生成 "assistant" 的起始 token (例如 <|im_start|>assistant\n)
                // 这样模型就知道该轮到它说话了
                inputs.add_generation_prompt = true; 

                // 生成最终 prompt
                current_prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
                
                // 调试打印：看看生成的 prompt 长什么样
                // LOG_INF("Formatted Chat Prompt:\n%s\n", current_prompt.c_str());

            } else {
                // [回退] 如果没有模板 (Base Model)，使用旧的拼接方式
                current_prompt = "<|im_start|>user\n"+params.prompt+"<|im_end|>\n";
            }
            // ============================================================
            // current_prompt = "<|im_start|>user\n"+params.prompt+"<|im_end|>\n";
            question_idx = 1;

            LOG_INF("\n\n=== Processing Single Prompt ===\n");
        }

        std::vector<int32_t> draft_skip_attn = default_skip_attn;
        std::vector<int32_t> draft_skip_mlp = default_skip_mlp;
        std::vector<int32_t> draft_layer_topk = default_layer_topk;
        int32_t draft_expert_topk = default_expert_topk;
        if (use_dataset) {
            std::vector<int32_t> parsed;
            if (extract_json_int_array(json_line, "draft_skip_attn", parsed)) {
                draft_skip_attn = parsed;
            }
            if (extract_json_int_array(json_line, "draft_skip_mlp", parsed)) {
                draft_skip_mlp = parsed;
            }
            if (extract_json_int_array(json_line, "draft_layer_topk", parsed)) {
                draft_layer_topk = parsed;
            }
            int32_t parsed_topk = -1;
            if (extract_json_int_value(json_line, "draft_expert_topk", parsed_topk)) {
                draft_expert_topk = parsed_topk;
            }
        }

        llama_context_set_draft_skip_layers(
            ctx_dft,
            draft_skip_attn.empty() ? nullptr : draft_skip_attn.data(),
            draft_skip_attn.size(),
            draft_skip_mlp.empty() ? nullptr : draft_skip_mlp.data(),
            draft_skip_mlp.size());
        llama_context_set_draft_expert_topk(ctx_dft, draft_expert_topk);
        llama_context_set_draft_layer_expert_topk(
            ctx_dft,
            draft_layer_topk.empty() ? nullptr : draft_layer_topk.data(),
            draft_layer_topk.size());
        // =============== [新增] 判断是否达到限制 ===============
        // // 如果设定了限制(>0)，且当前已处理数量(question_idx)达到了限制，则退出
        // if (params.n_questions_limit > 0 && question_idx >= params.n_questions_limit) {
        //     LOG_INF("\nReached question limit (%d). Stopping loop.\n", params.n_questions_limit);
        //     break;
        // }
        // if (json_line.empty()) continue;
        // question_idx++;

        // // 1. 解析 Question
        // std::string question = extract_json_value(json_line, "question");
        // if (question.empty()) continue;

        // // 构造 Prompt (你可以根据模型需求加上 Chat Template，例如 Qwen/Llama3 格式)
        // // 这里简单拼接，或者直接用 params.prompt 做前缀
        // std::string current_prompt = "Question: " + question + "\nAnswer:";
        
        // LOG_INF("\n\n=== Processing Question %d ===\n%s\n", question_idx, current_prompt.c_str());

                // 2. Tokenize (使用 current_prompt 替代 params.prompt)
        std::vector<llama_token> inp;
        inp = common_tokenize(ctx_tgt, current_prompt, true, true);
                // Tokenize the prompt
        // std::vector<llama_token> inp;
        // inp = common_tokenize(ctx_tgt, params.prompt, true, true);

        const int max_context_size     = llama_n_ctx(ctx_tgt);
        const int max_tokens_list_size = max_context_size - 4;

        if ((int) inp.size() > max_tokens_list_size) {
            LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
            // return 1;
            continue;
        }
        // // =======================================================
        // // [关键] 每次循环前必须清空 KV Cache
        // // =======================================================
        // llama_kv_cache_clear(ctx_tgt);
        // llama_kv_cache_clear(ctx_dft);
        // [关键] 每次循环前必须清空 KV Cache
        // =======================================================
        const bool share_kv = mem_tgt == mem_dft;
        llama_memory_clear(mem_tgt, true);
        if (!share_kv) {
            llama_memory_clear(mem_dft, true);
        }


        // 重置一些统计变量
        // int n_input = inp.size();
        int n_predict = 0;
        int n_drafted = 0;
        int n_accept  = 0;
        // int n_past_tgt = inp.size();
        // int n_past_dft = inp.size();
        // bool has_eos = false;
        LOG("\n\n");

        if (!use_search_stdin) {
            for (auto id : inp) {
                LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
            }
        }

        const int n_input = inp.size();

        const auto t_enc_start = ggml_time_us();

        // eval the prompt with both models
        // 1. Target 处理 Prompt (这会填充共享的 KV Cache)
        if (llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), n_input - 1)) != 0 ||
            llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(), 1)) != 0) {
            LOG_ERR("%s: target prompt decode failed, skipping question\n", __func__);
            continue;
        }

        // 2. Draft 处理 Prompt
        // === [修改] 注释掉下面这一行 ===
        // 在共享 KV 模式下，Target 已经填好了显存，Draft 直接用即可。
        // 如果不注释，会触发 discontinuous sequence 错误。
        // llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input));
        // 建议加上判断，为了代码兼容性：
        // 如果没有共享 (指针不同)，才执行 decode
        if (!share_kv) {
            if (llama_decode(ctx_dft, llama_batch_get_one(inp.data(), n_input)) != 0) {
                LOG_ERR("%s: draft prompt decode failed, skipping question\n", __func__);
                continue;
            }
        }
        const auto t_enc_end = ggml_time_us();

        // the 2 models should have the same vocab
        //GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

        const int n_draft_cap = params.speculative.n_max;
        MoeUtilityCascadeState cascade;
        cascade.init(params.speculative);
        int n_predict_mark = 0;

        int n_past_tgt = inp.size();
        int n_past_dft = inp.size();

        // used to determine end of generation
        bool has_eos = false;

        // target model sampling context (reuse the llama_context's sampling instance)
        struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

        // draft sequence data
        std::vector<seq_draft> drafts(n_seq_dft);

        for (int s = 0; s < n_seq_dft; ++s) {
            // allocate llama_sampler for each draft sequence
            params.sampling.temp=1.0f;
            drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
        }

        llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

        const auto t_dec_start = ggml_time_us();

        // sample from the last token of the prompt
        drafts[0].i_batch_tgt.resize(1);
        drafts[0].i_batch_tgt[0] = 0;
        int step_count = 0; // [新增] 记录步数
        const size_t question_trace_start = g_all_traces.size();
        while (true) {
            const int n_predict_before_step = n_predict;
            cascade.on_outer_iter_start(n_predict, n_predict_mark);
            const int n_draft_active = cascade.enabled ? cascade.active_k() : n_draft_cap;

            std::set<int> active_seqs = {};

            // print current draft sequences
            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].active) {
                    continue;
                }

                active_seqs.insert(s);
                const auto & tokens = drafts[s].tokens;

                LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
            }

            int i_dft  = 0;
            int s_keep = 0;

            llama_token token_id;
            std::string token_str;

            // loop until we fail to accept a drafted token or we run out of drafted tokens
            while (true) {
                //NvtxRange r_iter("iter");

                // check if the target token matches any of the drafts
                // for stochastic sampling, attempt to match the token with the drafted tokens
                {
                    NvtxRange r_verify("verify_loop"); 
                    bool accept = false;
                    if (params.sampling.temp > 0) {
                        // stochastic verification
                        common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);

                        auto & dist_tgt = *common_sampler_get_candidates(smpl, true);

                        float p_tgt = 0.0f;
                        float p_dft = 0.0f;

                        while (active_seqs.size() > 0) {
                            // randomly select a sequence to verify from active sequences
                            std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                            int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                            if (i_dft >= (int) drafts[s].tokens.size()) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                                continue;
                            }
                            if (accept) {
                                
                                // if we already accepted a token, we can skip the rest
                                if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                    drafts[s].active = false;
                                    active_seqs.erase(s);
                                }
                                continue;
                            }

                            LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int) active_seqs.size());
                            float r = u_dist(rng);
                            llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                            //GGML_ASSERT(dist_tgt.size <= dist_dft.size);

                            // acquire the token probabilities assigned by the draft and target models
                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                    p_tgt = dist_tgt.data[i].p;
                                    break;
                                }
                            }
                            for (size_t i = 0; i < dist_dft.size; i++) {
                                if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                    p_dft = dist_dft.data[i].p;
                                    break;
                                }
                            }
                            // [计算累积概率]
                            // 也就是 Draft 生成这个 token 时的 confidence * 之前所有 token 的 confidence
                            float cum_prob = 1.0f;
                            // 注意：i_dft 是当前验证到的位置。
                            // drafts[s].probs 存的是 Draft 对每个 token 的预测概率。
                            // 我们只乘到 i_dft 为止。
                            for (int k = 0; k <= i_dft && k < (int)drafts[s].probs.size(); ++k) {
                                cum_prob *= drafts[s].probs[k];
                            }
                            
                            // 获取当前单个 Token 的概率
                            float cur_prob = (i_dft < (int)drafts[s].probs.size()) ? drafts[s].probs[i_dft] : 0.0f;
                            LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);
                            if (r <= p_tgt / p_dft) {
                                s_keep = s;
                                accept = true;
                                token_id = drafts[s].tokens[i_dft];
                                token_str = common_token_to_piece(ctx_tgt, token_id);
                                common_sampler_accept(smpl, token_id, true);
                                // ================= [日志输出：接收] =================
                                // 使用 Cyan 色 (\033[36m) 
                                if (params.speculative.accept_log) {
                                    fprintf(stderr, "\033[36m[ANALYSIS] Token: '%s' | Prob: %.4f | CumProb: %.4f | Status: ACCEPT\033[0m\n",
                                            token_str.c_str(), cur_prob, cum_prob);
                                }

                                // ====================================================
                                LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                                log_draft_token_moe("accepted", true, s, i_dft, token_id, token_str, drafts[s]);
                                break;
                            } else {
                                const llama_token dft_token_id = drafts[s].tokens[i_dft];
                                const std::string dft_token_str = common_token_to_piece(ctx_tgt, dft_token_id);
                                LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, dft_token_id, dft_token_str.c_str());
                                log_draft_token_moe("rejected", false, s, i_dft, dft_token_id, dft_token_str, drafts[s]);
                                drafts[s].active = false;
                                // ================= [日志输出：拒绝] =================
                                // 使用 Red 色 (\033[31m)
                                if (params.speculative.accept_log) {
                                    fprintf(stderr, "\033[31m[ANALYSIS] Token: '%s' | Prob: %.4f | CumProb: %.4f | Status: REJECT\033[0m\n",
                                            common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str(), 
                                            cur_prob, cum_prob);
                                }
                                // ====================================================
                                // calculate residual probability
                                GGML_ASSERT(dist_tgt.sorted);
                                GGML_ASSERT(dist_dft.sorted);

                                // sort dist by id
                                std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                    return a.id < b.id;
                                });
                                std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                    return a.id < b.id;
                                });

                                float sum_probs = 0.0f;

                                for (size_t i = 0; i < dist_tgt.size; i++) {
                                    if (i < dist_dft.size) {
                                        dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                    } else {
                                        dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                    }

                                    sum_probs += dist_tgt.data[i].p;
                                }

                                for (size_t i = 0; i < dist_tgt.size; i++) {
                                    dist_tgt.data[i].p /= sum_probs;
                                }

                                // sort dist_tgt by p desc
                                std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                    return a.p > b.p;
                                });
                            }

                            active_seqs.erase(s);
                            for (int i = 0; i < n_seq_dft; i++) {
                                if (i == s) {
                                    continue;
                                }
                                if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                    // synchronize active status for sequences with the same drafted token
                                    drafts[i].active = drafts[i].active && accept;
                                    if (!drafts[i].active) {
                                        active_seqs.erase(s);
                                    }
                                }
                            }
                        }

                        if (!accept) {
                            // all drafted tokens were rejected
                            // sample from the target model
                            LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                            std::vector<float> probs(dist_tgt.size);
                            for (size_t i = 0; i < dist_tgt.size; ++i) {
                                probs[i] = dist_tgt.data[i].p;
                            }

                            std::discrete_distribution<> dist(probs.begin(), probs.end());

                            const int idx = dist(rng);

                            token_id = dist_tgt.data[idx].id;
                            common_sampler_accept(smpl, token_id, true);
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                        }
                    } else {
                        // greedy verification

                        // sample from the target model
                        LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                        token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);

                        common_sampler_accept(smpl, token_id, true);

                        token_str = common_token_to_piece(ctx_tgt, token_id);

                        for (int s = 0; s < n_seq_dft; ++s) {
                            if (!drafts[s].active) {
                                continue;
                            }

                            if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                                LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());

                                s_keep = s;
                                accept = true;
                                log_draft_token_moe("accepted", true, s, i_dft, token_id, token_str, drafts[s]);
                            } else {
                                if (i_dft < (int) drafts[s].tokens.size()) {
                                    const llama_token dft_token_id = drafts[s].tokens[i_dft];
                                    const std::string dft_token_str = common_token_to_piece(ctx_tgt, dft_token_id);
                                    log_draft_token_moe("rejected", false, s, i_dft, dft_token_id, dft_token_str, drafts[s]);
                                }
                                drafts[s].active = false;
                            }
                        }
                    }

                    if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                        has_eos = true;
                    }
                    ++n_predict;

                    if (accept) {
                        ++n_accept;
                        ++n_past_tgt;
                        ++n_past_dft;
                        // =========================================================
                        // [新增] Trace 更新: 标记节点被接受
                        // =========================================================
                        // s_keep 是当前被选中的 draft sequence ID
                        // i_dft 是当前被接受的 token 在 sequence 中的索引 (0-based)
                        // token_id 是被接受的 token ID
                        if (g_prune_config.enable_trace && g_all_traces.size() > question_trace_start) {
                            for (auto& node : g_all_traces.back().nodes) {
                                // 匹配条件：
                                // 1. 节点属于当前选中的序列 (node_id == s_keep)
                                // 2. Token ID 匹配
                                // 3. (可选但更严谨) 深度匹配: node.depth 对应 i_dft + 1 (因为 depth 从1开始)
                                if (node.node_id == s_keep && node.token_id == token_id) {
                                    node.accepted = true;
                                    break; // 找到了就跳出内层循环
                                }
                            }
                        }
                        // =========================================================
                        ++i_dft;
                        if (!use_search_stdin) {
                            if (params.use_color) {
                                // Color token according to its origin sequence
                                LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                            } else {
                                LOG("%s", token_str.c_str());
                            }
                        }
                        continue;
                    } else {
                        if (!use_search_stdin) {
                            LOG("%s", token_str.c_str());
                        }
                        break;
                    }
                }
            }
            if (g_prune_config.enable_trace && g_all_traces.size() > question_trace_start) {
                auto & completed_trace = g_all_traces.back();
                completed_trace.generated_tokens = n_predict - n_predict_before_step;
                completed_trace.effective_cost = 0;
                completed_trace.redundant_cost = 0;
                for (const auto & node : completed_trace.nodes) {
                    if (node.accepted) {
                        completed_trace.effective_cost += node.cost;
                    } else {
                        completed_trace.redundant_cost += node.cost;
                    }
                }
            }
            // ======================= [新增日志代码开始] =======================
            // i_dft 就是这一轮被接受的 token 数量
            // drafts[0].tokens.size() 大致是这一轮草稿生成的总长度
            // 使用 stderr 输出，配合颜色代码，方便在控制台一眼看到，不干扰正文生成
            if (n_predict > 0 && params.speculative.accept_log) { // 避免在刚开始处理 prompt 时打印
                int total_drafted = drafts[0].tokens.size(); // 草稿总长
                fprintf(stderr, "\n\033[1;33m[Speculative Stats] Drafted: %d | Accepted: %d | Rate: %.1f%%\033[0m", 
                        total_drafted, 
                        i_dft, 
                        total_drafted > 0 ? (100.0f * i_dft / total_drafted) : 0.0f);
            }
            // ======================= [新增日志代码结束] =======================
            {
                NvtxRange r_draft("draft_one step");
                LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

                // TODO: simplify
                {
                    LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                    llama_memory_seq_keep(mem_dft, s_keep);
                    llama_memory_seq_cp  (mem_dft, s_keep, 0, -1, -1);
                    llama_memory_seq_keep(mem_dft, 0);
                    // === [优化] 如果内存共享，不需要删两次 ===
                    if (mem_tgt != mem_dft) { 
                        llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);
                        llama_memory_seq_keep(mem_tgt, s_keep);
                        llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
                        llama_memory_seq_keep(mem_tgt, 0);
                    }
                    // llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);
                    // llama_memory_seq_keep(mem_tgt, s_keep);
                    // llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
                    // llama_memory_seq_keep(mem_tgt, 0);
                }

                for (int s = 0; s < n_seq_dft; ++s) {
                    drafts[s].active = false;
                    drafts[s].tokens.clear();
                    drafts[s].i_batch_tgt.clear();
                    drafts[s].dists.clear();
                    drafts[s].moe_layer_ratios.clear();
                    drafts[s].moe_ratio_avg.clear();
                    drafts[s].moe_layer_counts.clear();
                }
                // note: will be erased after the speculation phase
                drafts[0].tokens.push_back(token_id);
                // ================= [新增] =================
                // 这个 token 是 Target 生成的，对于 Draft 来说是已知条件，概率设为 1.0
                // 或者如果不参与计算累积概率，这一步其实不重要，但为了对齐 vector 长度必须加
                drafts[0].probs.push_back(1.0f); 
                // ==========================================
                drafts[0].dists.push_back(std::vector<llama_token_data>());
                drafts[0].i_batch_tgt.push_back(0);
                draft_moe_push_empty(drafts[0]);

                common_batch_clear(batch_dft);
                common_batch_add  (batch_dft, token_id, n_past_dft, { 0 }, true);

                llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
                // LOG_DBG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
                {
                    NvtxRange r_draft("draft_decode");
                    llama_decode(ctx_dft, batch_dft);
                }
                

                ++n_past_dft;
            }
            if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
                break;
            }

            if (drafts[0].smpl) {
                common_sampler_free(drafts[0].smpl);
            }
            drafts[0].smpl = common_sampler_clone(smpl);

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active   = false;
                drafts[s].drafting = false;
            }
            drafts[0].active      = true;
            drafts[0].drafting    = true;
            drafts[0].i_batch_dft = 0;

            common_batch_clear(batch_tgt);
            common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);
            // [新增] 创建当前步的 Trace 对象
            StepTrace current_trace;
            current_trace.step_idx = step_count++;
            if (params.speculative.prune == 0) {
                NvtxRange r_draft("draft_tree");
                draft_tree_original(
                    ctx_dft,
                    model_dft,
                    mem_dft,
                    batch_dft,
                    batch_tgt,
                    drafts,
                    n_seq_dft,
                    n_draft_active,
                    p_draft_split,
                    n_past_tgt,
                    n_past_dft,
                    n_drafted);
            } else {
                NvtxRange r_draft("adaptive_draft_tree");
                int MAX_DEPTH = params.speculative.prune_max_depth;
                int MAX_NODES = params.speculative.prune_max_nodes;
                MAX_DEPTH = std::max(1, MAX_DEPTH);
                MAX_NODES = std::max(1, MAX_NODES);
                MAX_NODES = std::max(1, std::min(MAX_NODES, std::max(1, n_draft_active)));

                draft_tree_prune(
                    ctx_dft,
                    model_dft,
                    mem_dft,
                    batch_dft,
                    batch_tgt,
                    drafts,
                    n_seq_dft,
                    n_draft_active,
                    n_past_tgt,
                    n_past_dft,
                    MAX_DEPTH,
                    MAX_NODES,
                    n_drafted,
                    current_trace);
            }
            // evaluate the target model on the drafted tokens
            {
                NvtxRange r_tgt("tgt_decode");
                llama_memory_seq_keep(mem_tgt, 0);
                for (int s = 1; s < n_seq_dft; ++s) {
                    llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
                }

                // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
                // ==========================================
                // [新增] 共享 KV Cache 的关键修正：逻辑回滚
                // ==========================================
                // Draft 刚刚写入了数据，导致 KV Cache 指针跑到了后面。
                // Target 需要从 n_past_tgt 开始验证（覆写）。
                // 所以我们需要把 n_past_tgt 之后的数据标记为“移除”（逻辑删除），
                // 这样 Target 就能从 n_past_tgt 开始写入了。
                
                // 判断是否共享了内存 (Self-Speculation)
                if (llama_get_memory(ctx_tgt) == llama_get_memory(ctx_dft)) {
                    // 对所有 sequence (-1)，清除从 n_past_tgt 开始的数据
                    llama_memory_seq_rm(mem_tgt, -1, n_past_tgt, -1);
                }
                // ==========================================
                llama_ssd_profile ssd_before = {};
                llama_ssd_profile ssd_after = {};
                llama_ssd_profile_snapshot(&ssd_before);
                auto t_start_verify = ggml_time_us();
                {
                    NvtxRange r_tgt("actual_verification_decode");
                    layer_profile_begin_step(current_trace.step_idx);
                    const bool reuse_verify = params.moe_reuse_runtime &&
                            params.moe_reuse_expert_cap != 0;
                    llama_context_set_moe_reuse_verify(ctx_tgt, reuse_verify, reuse_verify);
                    llama_decode(ctx_tgt, batch_tgt);
                    llama_context_set_moe_reuse_verify(ctx_tgt, false, false);
                    layer_profile_end_step();
                }
                auto t_end_verify = ggml_time_us();
                llama_ssd_profile_snapshot(&ssd_after);
                current_trace.t_target_ssd_fetch_us =
                    (double) (ssd_after.target_fetch_us - ssd_before.target_fetch_us);
                current_trace.target_ssd_read_requests =
                    ssd_after.target_read_requests - ssd_before.target_read_requests;
                current_trace.target_ssd_read_bytes =
                    ssd_after.target_read_bytes - ssd_before.target_read_bytes;
                update_target_routing_reuse_stats(
                    ctx_tgt,
                    batch_tgt.n_tokens,
                    resident_experts,
                    seen_target_experts,
                    routing_stats);
                current_trace.t_target_verify_us = (t_end_verify - t_start_verify); // 记录延迟
                total_t_verify_us += current_trace.t_target_verify_us;
                cascade.on_verify_done(t_end_verify - t_start_verify);
                llama_clear_ssd_cache_backend();
                // [新增] 填充 Target 信息
                // batch_tgt 中的 token 顺序对应 draft tree 中的节点顺序
                // 我们需要把 Target 的 Expert Info 和 Logits 填回 current_trace.nodes
                
                // 注意：batch_tgt 的构造逻辑需要和 trace nodes 对齐
                // 在 draft_tree_prune 里，节点是按 candidates 顺序加入的
                // 且 common_batch_add(batch_tgt, ...) 也是按这个顺序
                // 所以 trace.nodes[i] 对应 batch_tgt 的第 i+1 个位置 (第0个是 prompt 最后一个 token)
                // 或者是 draft 自己的映射关系。
                
                // 更准确的做法：遍历 current_trace.nodes，利用 node.node_id 找到 draft[s].i_batch_tgt
                // 这里为了简化，假设 batch_tgt 里的顺序就是 draft tree 遍历顺序
                if (g_prune_config.enable_trace) {
                    for (auto& node : current_trace.nodes) {
                        // 必须用创建节点时记录的槽位：同一 seq 在 prune 中会多次 push i_batch_tgt，
                        // drafts[s].i_batch_tgt.back() 只对最后一个节点正确，且可能与当前 batch 不一致，
                        // 会导致 llama_get_logits_ith 读到 logits 未启用的槽位（例如 invalid logits id 1）。
                        const int idx_in_batch = node.tmp_target_batch_idx;

                        node.target_experts = capture_experts(ctx_tgt, idx_in_batch);
                        node.target_logits_topk = get_logits_topk(ctx_tgt, idx_in_batch, 5);
                    }
                }
            

                ++n_past_tgt;
            }
            // 只有当这一步确实产生了 draft nodes 才记录，避免记录空 step
            if (!current_trace.nodes.empty()) {
                g_all_traces.push_back(current_trace);
                // LOG_INF("DEBUG: Trace step %d recorded with %zu nodes.\n", current_trace.step_idx, current_trace.nodes.size());
            } else {
                // 调试打印：确认是否因为为空被跳过
                // LOG_INF("DEBUG: Trace step %d is empty, skipping.\n", current_trace.step_idx);
            }
            // =========================================================
            
            // the first token is always proposed by the target model before the speculation loop so we erase it here
            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].active) {
                    continue;
                }

                drafts[s].tokens.erase(drafts[s].tokens.begin());
                // ================= [新增：同步删除概率] =================
                if (!drafts[s].probs.empty()) {
                    drafts[s].probs.erase(drafts[s].probs.begin());
                }
                // ======================================================
                drafts[s].dists.erase(drafts[s].dists.begin());
                if (!drafts[s].moe_layer_ratios.empty()) {
                    drafts[s].moe_layer_ratios.erase(drafts[s].moe_layer_ratios.begin());
                }
                if (!drafts[s].moe_ratio_avg.empty()) {
                    drafts[s].moe_ratio_avg.erase(drafts[s].moe_ratio_avg.begin());
                }
                if (!drafts[s].moe_layer_counts.empty()) {
                    drafts[s].moe_layer_counts.erase(drafts[s].moe_layer_counts.begin());
                }
            }
        }

        auto t_dec_end = ggml_time_us();
        // 5. 打印结果和清理当前问题的资源
        LOG("\n\n=== Answer %d ===\n", question_idx);
        // 如果你需要收集输出字符串，可以在 accept 发生时拼接到 string 里打印
        
        LOG_INF("Processed Question %d in %.3f s\n", question_idx, (t_dec_end - t_dec_start) / 1e6f);
        // 释放本次循环的资源
        // common_sampler_free(smpl);
        // for (int s = 0; s < n_seq_dft; ++s) {
        //     common_sampler_free(drafts[s].smpl);
        // }
        // llama_batch_free(batch_dft);
        // llama_batch_free(batch_tgt);

        LOG("\n\n");

        LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
        LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));
        if (use_search_stdin) {
            const double dec_sec = (t_dec_end - t_dec_start) / 1e6;
            const double tokens_per_s = dec_sec > 0.0 ? (n_predict / dec_sec) : 0.0;
            const double accept_rate = n_drafted > 0 ? (100.0 * n_accept / n_drafted) : 0.0;
            fprintf(stdout, "\nSEARCH_RESULT {\"tokens_per_s\":%.6f,\"accept_rate\":%.6f}\n", tokens_per_s, accept_rate);
            fflush(stdout);
        }

        if (!use_search_stdin) {
            LOG_INF("\n");
            LOG_INF("n_draft   = %d (cap; moe-utility-spec %s)\n", n_draft_cap, cascade.enabled ? "on" : "off");
            LOG_INF("n_predict = %d\n", n_predict);
            LOG_INF("n_drafted = %d\n", n_drafted);
            LOG_INF("n_accept  = %d\n", n_accept);
            LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);
            int verify_turns = n_predict - n_accept;
            LOG_INF("verify turns = %d\n", verify_turns);

            if (verify_turns > 0) {
                LOG_INF("accept length = %.3f\n", (double)n_accept  / (double)verify_turns);
                LOG_INF("draft  length = %.3f\n", (double)n_drafted / (double)verify_turns);
            } else {
                LOG_INF("accept length = inf (verify_turns=0)\n");
                LOG_INF("draft  length = inf (verify_turns=0)\n");
            }

            LOG_INF("\n");
            LOG_INF("draft:\n\n");
            // TODO: print sampling/grammar timings for all drafts
            llama_perf_context_print(ctx_dft);

            LOG_INF("\n");
            LOG_INF("target:\n\n");
            common_perf_print(ctx_tgt, smpl);
        }
        // ================= [修改] 累加统计数据，不打印单条详情 =================
        
        // 累加时间 (微秒)
        total_t_enc_us += (t_enc_end - t_enc_start);
        total_t_dec_us += (t_dec_end - t_dec_start);

        // 累加 Token 计数
        total_n_input   += n_input;
        total_n_predict += n_predict;
        total_n_drafted += n_drafted;
        total_n_accept  += n_accept;
        processed_count++;

        // 可选：打印一个简单的进度条，让你知道程序还活着
        fprintf(stderr, "\rProcessed: %d questions | Last accepted rate: %.1f%%\n", 
                processed_count, (n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f));


        common_sampler_free(smpl);
        for (int s = 0; s < n_seq_dft; ++s) {
            common_sampler_free(drafts[s].smpl);
        }



        llama_batch_free(batch_dft);
        llama_batch_free(batch_tgt);

        // 强制刷新日志输出
        fflush(stdout); 
    }
    // ================= [新增] 计算并打印最终平均值 =================
    if (!use_search_stdin) {
        LOG("\n\n==================== FINAL STATISTICS ====================\n");
        LOG("Total Questions Processed: %d\n", processed_count);

        if (processed_count > 0) {
            // 转换时间为秒
            double total_enc_sec = total_t_enc_us / 1e6;
            double total_dec_sec = total_t_dec_us / 1e6;

            // 计算整体速度 (Total Tokens / Total Time)
            // 这样比 "求速度的平均值" 更能反映整体吞吐量
            double avg_enc_speed = total_n_input / total_enc_sec;
            double avg_dec_speed = total_n_predict / total_dec_sec;
            
            // 计算整体接受率
            double avg_accept_rate = (total_n_drafted > 0) ? (100.0 * total_n_accept / total_n_drafted) : 0.0;
            double overall_acceptance_length = (total_n_predict - total_n_accept) > 0 ? 
                ((double)total_n_accept / (double)(total_n_predict - total_n_accept)) : 0.0;

            LOG_INF("Total Input Tokens:    %lld\n", total_n_input);
            LOG_INF("Total Gen Tokens:      %lld\n", total_n_predict);
            LOG_INF("Total Encoding Time:   %.3f s\n", total_enc_sec);
            LOG_INF("Total Decoding Time:   %.3f s\n", total_dec_sec);
            LOG("\n");
            LOG_INF("Avg Encoding Speed:    %.3f t/s\n", avg_enc_speed);
            LOG_INF("Avg Decoding Speed:    %.3f t/s\n", avg_dec_speed);
            LOG("\n");
            LOG_INF("Total Drafted:         %lld\n", total_n_drafted);
            LOG_INF("Total Accepted:        %lld\n", total_n_accept);
            LOG_INF("Overall Accept Rate:   %.3f%%\n", avg_accept_rate);
            LOG_INF("Overall Accept Length: %.3f\n", overall_acceptance_length);
        }
        llama_ssd_profile ssd_profile = {};
        llama_ssd_profile_snapshot(&ssd_profile);
        LOG(
            "S2MOE_PROFILE_JSON {\"processed_questions\": %d, "
            "\"generated_tokens\": %lld, \"accepted_tokens\": %lld, "
            "\"drafted_tokens\": %lld, \"target_verify_us\": %.0f, "
            "\"ssd_read_bytes\": %llu, \"ssd_read_requests\": %llu, "
            "\"ssd_fetch_us\": %llu, \"ssd_resident_hits\": %llu, "
            "\"ssd_misses\": %llu, \"ssd_expert_accesses\": %llu, "
            "\"ssd_draft_read_bytes\": %llu, \"ssd_draft_read_requests\": %llu, "
            "\"ssd_draft_fetch_us\": %llu, \"ssd_target_read_bytes\": %llu, "
            "\"ssd_target_read_requests\": %llu, \"ssd_target_fetch_us\": %llu, "
            "\"ssd_prefill_read_bytes\": %llu, \"ssd_prefill_read_requests\": %llu, "
            "\"ssd_prefill_fetch_us\": %llu, "
            "\"routing_verify_steps\": %lld, "
            "\"target_activated_experts\": %lld, "
            "\"target_unique_experts\": %lld, "
            "\"target_batch_reuse_hits\": %lld, "
            "\"target_new_routed_experts\": %lld, "
            "\"target_resident_routed_hits\": %lld, "
            "\"target_offchip_routed_experts\": %lld}\n",
            processed_count,
            total_n_predict,
            total_n_accept,
            total_n_drafted,
            total_t_verify_us,
            (unsigned long long) ssd_profile.read_bytes,
            (unsigned long long) ssd_profile.read_requests,
            (unsigned long long) ssd_profile.fetch_us,
            (unsigned long long) ssd_profile.resident_hits,
            (unsigned long long) ssd_profile.ssd_misses,
            (unsigned long long) ssd_profile.expert_accesses,
            (unsigned long long) ssd_profile.draft_read_bytes,
            (unsigned long long) ssd_profile.draft_read_requests,
            (unsigned long long) ssd_profile.draft_fetch_us,
            (unsigned long long) ssd_profile.target_read_bytes,
            (unsigned long long) ssd_profile.target_read_requests,
            (unsigned long long) ssd_profile.target_fetch_us,
            (unsigned long long) ssd_profile.prefill_read_bytes,
            (unsigned long long) ssd_profile.prefill_read_requests,
            (unsigned long long) ssd_profile.prefill_fetch_us,
            routing_stats.verify_steps,
            routing_stats.target_activated_experts,
            routing_stats.target_unique_experts,
            routing_stats.target_batch_reuse_hits,
            routing_stats.target_new_routed_experts,
            routing_stats.target_resident_routed_hits,
            routing_stats.target_offchip_routed_experts);
        log_ratio_distribution();
        LOG("==========================================================\n\n");
    }
    // ===============================================================
    if (g_prune_config.enable_trace) {
        save_trace_to_json(g_trace_filename);
    }
    save_layer_profile_json();

    llama_backend_free();
    LOG("\n\n");

    return 0;
}
