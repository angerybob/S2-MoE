#include "common.h"
#include "llama.h"
#include <vector>
#include <cstdio>
#include <algorithm>
#include <string>
#include <cmath>
#include <cassert>
#include <map>
#include <random>
#include <chrono>

// 时间工具
static uint64_t get_time_ns() {
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

// 性能统计结构
struct BenchStats {
    uint64_t total_time_ns = 0;
    uint64_t prompt_time_ns = 0;
    uint64_t gen_time_ns = 0;
    int total_tokens = 0;
    int prompt_tokens = 0;
    int gen_tokens = 0;
    int total_drafted = 0;
    int total_accepted = 0;
    int prob_accepts = 0;
    int verify_steps = 0;

    double prompt_tps = 0.0;
    double gen_tps = 0.0;
    double overall_tps = 0.0;
    double acceptance_rate = 0.0;
    double mean_spec_len = 0.0;
};

// ==================== 必要的 API 声明 ====================
extern "C" float * llama_context_get_eagle_hs_at_batch_idx(struct llama_context * ctx, int batch_idx, int32_t * out_dim);

// ==================== 配置常量 ====================
const int MAX_TARGET_FORWARDS = 256;   // Target forward次数（与单链版本对齐）
const int TOP_K_L1 = 3;               // 第一步 Top K (简化为只做一步)
const int TOTAL_LANES = 3;            // 3 个 Lane 刚好对应 3 条路径

// ==================== 全局随机工具 ====================
// 使用固定种子确保结果可重现
static std::mt19937 g_rng(42);  // 固定种子42
static std::uniform_real_distribution<float> g_uniform_dist(0.0f, 1.0f);

// ==================== 辅助结构与函数 ====================

struct DraftPath {
    int lane_id;             // 0 ~ 7
    llama_token t1;          // 第一步预测
    float prob_t1;
    int rank;                // 在 top-k 中的排名 (0-7)
};

// 统计
struct Stats {
    int total_accepts = 0;
    int total_drafted = 0;
    int verify_steps = 0;
    int prob_accepts = 0;  // 统计概率性接收次数
    std::vector<int> rank_accepts; // 统计每个 rank 的接收次数
} g_stats;

// 辅助：获取贪婪预测 - 修复Logits索引错位
llama_token get_greedy_token(llama_context* ctx, int batch_idx) {
    // 使用 llama_get_logits_ith 精确获取指定 batch index 的 logits
    const float* current_logits = llama_get_logits_ith(ctx, batch_idx);
    if (!current_logits) {
        printf("[ERROR] No logits for batch_idx %d\n", batch_idx);
        return 0;
    }

    const auto* vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n_vocab = llama_vocab_n_tokens(vocab);

    int best_id = 0;
    float max_val = -1e9;
    for (int i = 0; i < n_vocab; i++) {
        if (current_logits[i] > max_val) {
            max_val = current_logits[i];
            best_id = i;
        }
    }
    return best_id;
}

// 辅助：获取特定token的概率 - 修复Logits索引错位
float get_token_probability(llama_context* ctx, int batch_idx, llama_token token) {
    // 使用 llama_get_logits_ith 精确获取指定 batch index 的 logits
    const float* current_logits = llama_get_logits_ith(ctx, batch_idx);
    if (!current_logits) {
        printf("[ERROR] No logits for batch_idx %d in get_token_probability\n", batch_idx);
        return 0.0f;
    }

    const auto* vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n_vocab = llama_vocab_n_tokens(vocab);

    float max_logit = current_logits[0];
    for (int i = 1; i < n_vocab; i++) if (current_logits[i] > max_logit) max_logit = current_logits[i];

    float sum_exp = 0.0f;
    float token_exp = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        float exp_val = expf(current_logits[i] - max_logit);
        sum_exp += exp_val;
        if (i == token) token_exp = exp_val;
    }
    return token_exp / sum_exp;
}

// 辅助：获取 Top K - 返回 softmax 概率而不是 logits
std::vector<std::pair<llama_token, float>> get_top_k_tokens_with_logits(llama_context* ctx, int batch_idx, int k) {
    // 使用 llama_get_logits_ith 精确获取指定 batch index 的 logits
    const float* current_logits = llama_get_logits_ith(ctx, batch_idx);
    if (!current_logits) {
        printf("[ERROR] No logits for batch_idx %d in get_top_k_tokens_with_logits\n", batch_idx);
        return {};
    }

    const auto* vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n_vocab = llama_vocab_n_tokens(vocab);

    // 1. Find Max Logit for numerical stability
    float max_logit = current_logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (current_logits[i] > max_logit) {
            max_logit = current_logits[i];
        }
    }

    // 2. Compute Softmax Probabilities (Full Vocabulary)
    std::vector<std::pair<llama_token, float>> all_tokens;
    all_tokens.reserve(n_vocab);
    float sum_exp = 0.0f;

    // First pass: compute sum of exponentials
    for (int i = 0; i < n_vocab; i++) {
        float exp_val = expf(current_logits[i] - max_logit);
        sum_exp += exp_val;
    }

    // Second pass: compute normalized probabilities
    for (int i = 0; i < n_vocab; i++) {
        float exp_val = expf(current_logits[i] - max_logit);
        float prob = exp_val / sum_exp;
        all_tokens.emplace_back(i, prob);
    }

    // 3. Select Top K by probability
    std::partial_sort(all_tokens.begin(), all_tokens.begin() + k, all_tokens.end(),
        [](const std::pair<llama_token, float>& a, const std::pair<llama_token, float>& b) {
            return a.second > b.second;
        });

    return std::vector<std::pair<llama_token, float>>(all_tokens.begin(), all_tokens.begin() + k);
}

// 核心：使用新 API 提取 HS
std::vector<float> extract_hs(llama_context* ctx, int batch_idx) {
    int32_t dims;
    float* ptr = llama_context_get_eagle_hs_at_batch_idx(ctx, batch_idx, &dims);

    if (!ptr) {
        printf("[ERROR] extract_hs failed for batch_idx %d\n", batch_idx);
        return {};
    }

    std::vector<float> res(ptr, ptr + dims);
    free(ptr); // 释放 C API 分配的内存
    return res;
}

// 辅助：添加单个 token 到 batch (单 Lane)
void add_lane_token(llama_batch& batch, llama_token token, int pos, int lane_id, bool logits) {
    int idx = batch.n_tokens;
    batch.token[idx] = token;
    batch.pos[idx] = pos;
    batch.n_seq_id[idx] = 1;
    batch.seq_id[idx][0] = lane_id;
    batch.logits[idx] = logits;
    batch.n_tokens++;
}

// 核心逻辑：EAGLE 概率接收判断
bool check_eagle_acceptance(float p_target, float p_draft, llama_token candidate, llama_token target_greedy) {
    // 1. 如果是 Greedy Match，直接接受
    if (candidate == target_greedy) return true;

    // 2. 概率接收 (Rejection Sampling)
    if (p_draft <= 0.0f) return false; // 防止除零

    float r = p_target / p_draft;
    float random_val = g_uniform_dist(g_rng);

    if (random_val < r) {
        return true;
    }
    return false;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <tgt> <dft> [reps]\n", argv[0]);
        return 1;
    }

    int reps = argc > 3 ? atoi(argv[3]) : 5;
    if (reps < 1) reps = 1;

    printf("EAGLE3 Tree Benchmark: %d repetitions\n", reps);
    printf("Target model: %s\n", argv[1]);
    printf("Draft model:  %s\n", argv[2]);
    printf("Tree config: TOP_K_L1=%d, TOTAL_LANES=%d (simplified 1-step)\n", TOP_K_L1, TOTAL_LANES);

    std::vector<BenchStats> results;

    common_init();
    llama_backend_init();

    // 1. 模型加载 (load once, reuse for all runs)
    llama_model_params mp = llama_model_default_params();
    llama_model* m_tgt = llama_model_load_from_file(argv[1], mp);
    llama_model* m_dft = llama_model_load_from_file(argv[2], mp);

    // 2. 上下文配置
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 4096;          // 确保足够大，容纳 Prompt + 生成 + SMP 并行开销
    cp.n_batch = 512;
    cp.n_seq_max = TOTAL_LANES; // 必须 >= 16
    cp.embeddings = false;

    llama_context* ctx_tgt = llama_init_from_model(m_tgt, cp);
    llama_context* ctx_dft = llama_init_from_model(m_dft, cp);

    // EAGLE Setup
    llama_context_set_draft_context(ctx_dft, true);
    llama_context_set_target_embedding_layer(ctx_dft, ctx_tgt);
    const int32_t* d2t_map = llama_model_get_eagle_d2t_map(m_dft);

    // Create CPU target model and context for embeddings only
    llama_model_params mp_cpu = llama_model_default_params();
    mp_cpu.n_gpu_layers = 0;     // Force CPU-only loading
    llama_model* m_tgt_cpu = llama_model_load_from_file(argv[1], mp_cpu);

    llama_context_params cp_cpu = llama_context_default_params();
    cp_cpu.n_ctx = 2048;
    cp_cpu.embeddings = true;    // Need embeddings
    cp_cpu.n_batch = 1;         // Minimal batch for single token
    llama_context* ctx_tgt_cpu = llama_init_from_model(m_tgt_cpu, cp_cpu);

    // Prompt
    std::string prompt = "The future of artificial intelligence and machine learning is incredibly exciting. We are seeing rapid advances in natural language processing, computer vision, and reinforcement learning.";
    std::vector<llama_token> tokens = common_tokenize(ctx_tgt, prompt, false);
    printf("Prompt length: %zu tokens\n", tokens.size());

    // 初始化 rank 统计
    g_stats.rank_accepts.resize(TOP_K_L1, 0);

    // Warmup run
    printf("Warmup...\n");

    // Benchmark runs
    for (int rep = 0; rep < reps; rep++) {
        printf("Run %d/%d...\n", rep + 1, reps);
        fflush(stdout);

        // Reset statistics
        g_stats = {0};
        g_stats.rank_accepts.resize(TOP_K_L1, 0);
        BenchStats stats;
        uint64_t total_start = get_time_ns();

        // 重置模型状态
        llama_memory_clear(llama_get_memory(ctx_tgt), false);
        llama_memory_clear(llama_get_memory(ctx_dft), false);

        // --- Prompt Phase ---
        uint64_t prompt_start = get_time_ns();

        // 3.1 Lane 0 Decode
        llama_batch prompt_batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); ++i) {
            prompt_batch.token[i] = tokens[i];
            prompt_batch.pos[i] = i;
            prompt_batch.n_seq_id[i] = 1;
            prompt_batch.seq_id[i][0] = 0; // 只给 Lane 0
            prompt_batch.logits[i] = (i == tokens.size() - 1);
            prompt_batch.n_tokens++;
        }

        if (llama_decode(ctx_tgt, prompt_batch) != 0) {
            fprintf(stderr, "Prompt decode failed\n");
            llama_batch_free(prompt_batch);
            continue;
        }
        llama_batch_free(prompt_batch);

        // 3.2 Sync: Copy Lane 0 to 1..15
        llama_memory_t mem = llama_get_memory(ctx_tgt);
        for (int i = 1; i < TOTAL_LANES; ++i) {
            llama_memory_seq_cp(mem, 0, i, -1, -1);
        }

        uint64_t prompt_end = get_time_ns();
        stats.prompt_time_ns = prompt_end - prompt_start;

        // --- State Initialization ---
        // 4. 初始化状态
        // Prompt Batch 中最后一个 token (Index = tokens.size()-1) 产生了 HS
        std::vector<float> current_hs = extract_hs(ctx_tgt, tokens.size() - 1);
        llama_token next_ground_truth = get_greedy_token(ctx_tgt, tokens.size() - 1);

        if (current_hs.empty()) {
            fprintf(stderr, "[FATAL] Failed to extract initial HS. Check API implementation.\n");
            continue;
        }

        std::vector<llama_token> generated = tokens;

        // --- Generation Phase ---
        uint64_t gen_start = get_time_ns();

        // ==================== 主循环 ====================
        for (int target_forward_count = 0; target_forward_count < MAX_TARGET_FORWARDS; target_forward_count++) {
            int current_pos = generated.size();

            // --- Phase 1: Single-Step Draft ---
            // Root -> Top 8 (L1)

            // 清空 Draft 的旧 KV cache
            llama_memory_clear(llama_get_memory(ctx_dft), false);

            llama_context_set_target_hidden_states(ctx_dft, current_hs.data(), current_hs.size());

            // CPU embedding computation for dual-model loading
            llama_context_compute_token_embedding_cpu(ctx_dft, ctx_tgt_cpu, next_ground_truth);

            llama_batch dft_batch = llama_batch_get_one(&next_ground_truth, 1);
            llama_decode(ctx_dft, dft_batch);

            auto top_k_l1 = get_top_k_tokens_with_logits(ctx_dft, 0, TOP_K_L1);

            // 收集所有路径 (8 条路径)
            std::vector<DraftPath> paths;
            for(int i = 0; i < TOP_K_L1; ++i) {
                llama_token t1_dft = top_k_l1[i].first;
                llama_token t1_tgt = t1_dft + d2t_map[t1_dft];
                paths.push_back({i, t1_tgt, top_k_l1[i].second, i}); // rank = i
            }
            g_stats.total_drafted += paths.size(); // 每个路径1个token

            // --- Phase 2: Verify Batch (8 Lanes, 2 tokens each) ---
            // 每个 Lane: [Root (P), T1 (P+1)]

            int batch_size = paths.size() * 2; // 8 * 2 = 16
            llama_batch v_batch = llama_batch_init(batch_size, 0, TOTAL_LANES);

            // 记录 Batch Index 用于提取
            struct LaneIndices { int root; int t1; };
            std::map<int, LaneIndices> lane_indices;

            for(const auto& path : paths) {
                int l = path.lane_id;

                // Root (P)
                add_lane_token(v_batch, next_ground_truth, current_pos, l, true); // 需要 Root Logits
                int idx_root = v_batch.n_tokens - 1;

                // T1 (P+1)
                add_lane_token(v_batch, path.t1, current_pos + 1, l, true); // 需要 T1 Logits
                int idx_t1 = v_batch.n_tokens - 1;

                lane_indices[l] = {idx_root, idx_t1};
            }

            if (llama_decode(ctx_tgt, v_batch) != 0) {
                printf("Decode failed\n");
                llama_batch_free(v_batch);
                break;
            }

            // --- Phase 3: Elect Winner (Probabilistic Verification) ---

            // 共享 Root Logits，随便取一个 lane (e.g., lane 0) 的 root logits
            int root_logits_idx = lane_indices[0].root;
            llama_token root_greedy_prediction = get_greedy_token(ctx_tgt, root_logits_idx);

            int accepted_rank = -1;
            llama_token accepted_token = 0;
            int best_lane = -1;

            // 遍历 L1 候选者 (按 Draft 概率从高到低，即按 rank 排序)
            for(int i = 0; i < TOP_K_L1; ++i) {
                llama_token candidate = top_k_l1[i].first + d2t_map[top_k_l1[i].first];
                float p_draft = top_k_l1[i].second;
                float p_target = get_token_probability(ctx_tgt, root_logits_idx, candidate);

                bool accepted = check_eagle_acceptance(p_target, p_draft, candidate, root_greedy_prediction);

                if (accepted) {
                    if (candidate != root_greedy_prediction) g_stats.prob_accepts++;

                    accepted_rank = i;
                    accepted_token = candidate;
                    best_lane = i; // rank 和 lane_id 相同
                    g_stats.rank_accepts[i]++; // 记录这个 rank 被接受
                    break; // 找到第一个接受的即停止
                }
            }

            // --- Phase 4: Merge & Sync ---
            generated.push_back(next_ground_truth); // Root Always Accepted

            if (accepted_rank != -1) {
                // Hit: 接受了某个 token
                auto idx = lane_indices[best_lane];

                g_stats.total_accepts += 1;
                generated.push_back(accepted_token);

                // Extract from T1 pos (Pos+1)
                current_hs = extract_hs(ctx_tgt, idx.t1);
                next_ground_truth = get_greedy_token(ctx_tgt, idx.t1);

                // Sync [P, P+2) - 全部清除然后完整复制
                llama_memory_t mem_sync = llama_get_memory(ctx_tgt);
                for(int i=0; i<TOTAL_LANES; ++i) {
                    if(i != best_lane) {
                        llama_memory_seq_rm(mem_sync, i, -1, -1);
                        llama_memory_seq_cp(mem_sync, best_lane, i, -1, -1);
                    }
                }

            } else {
                // Miss: 没有接受任何 token
                auto idx = lane_indices[0]; // Any lane works for Root

                current_hs = extract_hs(ctx_tgt, idx.root);
                next_ground_truth = root_greedy_prediction;

                // Sync [P, P+1) - 全部清除然后完整复制
                llama_memory_t mem_miss = llama_get_memory(ctx_tgt);
                for(int i=1; i<TOTAL_LANES; ++i) {
                    llama_memory_seq_rm(mem_miss, i, -1, -1);
                    llama_memory_seq_cp(mem_miss, 0, i, -1, -1);
                }
            }

            // --- Phase 5: Trash Collection ---
            int keep_len = (accepted_rank != -1) ? 2 : 1;
            llama_memory_t mem = llama_get_memory(ctx_tgt);
            llama_memory_seq_rm(mem, -1, current_pos + keep_len, -1);

            g_stats.verify_steps++;
            llama_batch_free(v_batch);
        }

        // --- Generation Phase End ---
        uint64_t gen_end = get_time_ns();
        stats.gen_time_ns = gen_end - gen_start;
        stats.total_time_ns = gen_end - total_start;

        // 填充统计数据
        stats.total_tokens = generated.size();
        stats.prompt_tokens = tokens.size();
        stats.gen_tokens = generated.size() - tokens.size();
        stats.total_drafted = g_stats.total_drafted;
        stats.total_accepted = g_stats.total_accepts;
        stats.prob_accepts = g_stats.prob_accepts;
        stats.verify_steps = g_stats.verify_steps;  // 这应该等于MAX_TARGET_FORWARDS

        // 计算性能指标
        stats.prompt_tps = stats.prompt_tokens / (stats.prompt_time_ns / 1e9);
        stats.gen_tps = stats.gen_tokens / (stats.gen_time_ns / 1e9);
        stats.overall_tps = stats.total_tokens / (stats.total_time_ns / 1e9);
        stats.acceptance_rate = stats.total_drafted > 0 ? (double)stats.total_accepted / stats.total_drafted : 0.0;
        stats.mean_spec_len = stats.verify_steps > 0 ? (double)stats.total_accepted / stats.verify_steps : 0.0;

        results.push_back(stats);
        fflush(stdout);
    }

    // 打印结果
    if (results.empty()) {
        printf("No successful runs\n");
    } else {
        // 计算平均值
        BenchStats avg;
        for (const auto& r : results) {
            avg.total_time_ns += r.total_time_ns;
            avg.prompt_time_ns += r.prompt_time_ns;
            avg.gen_time_ns += r.gen_time_ns;
            avg.total_tokens += r.total_tokens;
            avg.prompt_tokens += r.prompt_tokens;
            avg.gen_tokens += r.gen_tokens;
            avg.total_drafted += r.total_drafted;
            avg.total_accepted += r.total_accepted;
            avg.prob_accepts += r.prob_accepts;
            avg.verify_steps += r.verify_steps;
        }

        size_t n = results.size();
        avg.total_time_ns /= n;
        avg.prompt_time_ns /= n;
        avg.gen_time_ns /= n;
        avg.total_tokens /= n;
        avg.prompt_tokens /= n;
        avg.gen_tokens /= n;
        avg.total_drafted /= n;
        avg.total_accepted /= n;
        avg.prob_accepts /= n;
        avg.verify_steps /= n;

        avg.prompt_tps = avg.prompt_tokens / (avg.prompt_time_ns / 1e9);
        avg.gen_tps = avg.gen_tokens / (avg.gen_time_ns / 1e9);
        avg.overall_tps = avg.total_tokens / (avg.total_time_ns / 1e9);
        avg.acceptance_rate = avg.total_drafted > 0 ? (double)avg.total_accepted / avg.total_drafted : 0.0;
        avg.mean_spec_len = avg.verify_steps > 0 ? (double)avg.total_accepted / avg.verify_steps : 0.0;

        printf("\n=== EAGLE3 Tree Benchmark Results (%d Target Forwards) ===\n", MAX_TARGET_FORWARDS);
        printf("Tree Config: TOP_K_L1=%d, TOTAL_LANES=%d (1-step parallel)\n", TOP_K_L1, TOTAL_LANES);
        printf("Prompt:     %d tokens, %.2f TPS\n", avg.prompt_tokens, avg.prompt_tps);
        printf("Generation: %d tokens, %.2f TPS\n", avg.gen_tokens, avg.gen_tps);
        printf("Overall:    %d tokens, %.2f TPS\n", avg.total_tokens, avg.overall_tps);
        printf("Target Forwards: %d, Generated/Forward: %.2f\n",
               avg.verify_steps, avg.gen_tokens / (double)avg.verify_steps);
        printf("Acceptance Rate: %.2f%%, Mean Spec Length: %.2f\n",
               avg.acceptance_rate * 100, avg.mean_spec_len);
        printf("Prob Accepts: %d/%d (%.1f%% of accepts)\n",
               avg.prob_accepts, avg.total_accepted,
               avg.total_accepted > 0 ? (double)avg.prob_accepts / avg.total_accepted * 100 : 0.0);

        // 打印 rank 接收分布
        printf("\n--- Rank Acceptance Distribution ---\n");
        for (int i = 0; i < TOP_K_L1; ++i) {
            printf("Rank %d: %d accepts (%.1f%%)\n", i, g_stats.rank_accepts[i],
                   g_stats.total_accepts > 0 ? (double)g_stats.rank_accepts[i] / g_stats.total_accepts * 100 : 0.0);
        }
    }

    llama_free(ctx_tgt); llama_free(ctx_tgt_cpu); llama_free(ctx_dft);
    llama_model_free(m_tgt); llama_model_free(m_tgt_cpu); llama_model_free(m_dft);
    llama_backend_free();
    return 0;
}