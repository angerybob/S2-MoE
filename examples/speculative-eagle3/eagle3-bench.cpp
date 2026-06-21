#include "common.h"
#include "llama.h"
#include <vector>
#include <cstdio>
#include <algorithm>
#include <string>
#include <cmath>
#include <random>
#include <map>
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

// Forward declarations
llama_token get_greedy_token(llama_context* ctx, int batch_idx);
std::vector<float> extract_hs(llama_context* ctx, int batch_idx);

// === 配置区域 ===
const int MAX_TARGET_FORWARDS = 256;   // Target forward次数（不再是token数量）
const int SPECULATIVE_DEPTH = 1;      // 每次 Draft 预测 K 步

// 全局随机工具
// 使用固定种子确保结果可重现
static std::mt19937 g_rng(42);  // 固定种子42
static std::uniform_real_distribution<float> g_uniform_dist(0.0f, 1.0f);

// 统计
struct Stats {
    int total_accepts = 0;
    int total_drafted = 0;
    int verify_steps = 0;
    int prob_accepts = 0;  // 统计概率性接收次数
} g_stats;

// 辅助：获取贪婪预测
llama_token get_greedy_token(llama_context* ctx, int batch_idx) {
    auto* logits = llama_get_logits(ctx);
    const auto* vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n_vocab = llama_vocab_n_tokens(vocab);
    // 偏移 logits 指针到 batch 中的特定位置
    const float* current_logits = logits + (batch_idx * n_vocab);

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

// 辅助：获取特定token的概率 - 使用与get_greedy_token相同的GPU兼容logits访问方式
float get_token_probability(llama_context* ctx, int batch_idx, llama_token token) {
    auto* logits = llama_get_logits(ctx);
    const auto* vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n_vocab = llama_vocab_n_tokens(vocab);
    // 使用与get_greedy_token完全相同的偏移方式
    const float* current_logits = logits + (batch_idx * n_vocab);

    // 找到最大logit用于数值稳定性
    float max_logit = current_logits[0];
    for (int i = 1; i < n_vocab; i++) if (current_logits[i] > max_logit) max_logit = current_logits[i];

    // 计算softmax概率
    float sum_exp = 0.0f;
    float token_exp = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        float exp_val = expf(current_logits[i] - max_logit);
        sum_exp += exp_val;
        if (i == token) token_exp = exp_val;
    }
    return token_exp / sum_exp;
}

// 核心：EAGLE 概率接收判断
// 遵循公式: r = min(1, P_target / P_draft)
bool check_eagle_acceptance(float p_target, float p_draft, llama_token candidate, llama_token target_greedy) {
    // 1. 如果是 Greedy Match，直接接受
    if (candidate == target_greedy) return true;

    // 2. 概率接收 (Rejection Sampling) - 这里的关键是 target_prob >= draft_prob * 1.0f
    if (p_draft <= 0.0f) return false; // 防止除零

    float r = p_target / p_draft;  // 这实现了 target_prob >= draft_prob * 1.0f 的逻辑
    float random_val = g_uniform_dist(g_rng);

    if (random_val < r) {
        return true;
    }
    return false;
}

// 辅助：从当前batch中提取指定token的HS (使用新API)
// 这个函数需要在每次decode后立即调用，因为底层buffer只保留当前batch的数据
std::vector<float> extract_hs(llama_context* ctx, int batch_idx) {
    std::vector<float> res;

    // 使用新的API提取指定batch_idx的EAGLE hidden states
    int32_t dims;
    float* token_hs = llama_context_get_eagle_hs_at_batch_idx(ctx, batch_idx, &dims);

    if (!token_hs) {
        printf("[ERROR] Failed to extract HS for batch_idx %d\n", batch_idx);
        return {};
    }

    // 直接复制返回，新API已经返回了所有EAGLE层的完整数据
    res.assign(token_hs, token_hs + dims);
    free(token_hs);

    return res;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);

    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <tgt> <dft> [reps]\n", argv[0]);
        return 1;
    }

    int reps = argc > 3 ? atoi(argv[3]) : 5;
    if (reps < 1) reps = 1;

    printf("EAGLE3 Benchmark: %d repetitions\n", reps);
    printf("Target model: %s\n", argv[1]);
    printf("Draft model:  %s\n", argv[2]);

    std::vector<BenchStats> results;

    common_init();
    llama_backend_init();

    // Load Models (load once, reuse for all runs)
    llama_model_params mp = llama_model_default_params();
    llama_model* m_tgt = llama_model_load_from_file(argv[1], mp);
    llama_model* m_dft = llama_model_load_from_file(argv[2], mp);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 512; // Batch必须足够大以容纳 verification batch

    cp.embeddings = true; // Target need HS (pre-attention hidden states for EAGLE3)
    llama_context* ctx_tgt = llama_init_from_model(m_tgt, cp);
    cp.embeddings = false;
    llama_context* ctx_dft = llama_init_from_model(m_dft, cp);

    // Create CPU target model and context for embeddings only
    llama_model_params mp_cpu = llama_model_default_params();
    mp_cpu.n_gpu_layers = 0;     // Force CPU-only loading
    llama_model* m_tgt_cpu = llama_model_load_from_file(argv[1], mp_cpu);

    llama_context_params cp_cpu = llama_context_default_params();
    cp_cpu.n_ctx = 2048;
    cp_cpu.embeddings = true;    // Need embeddings
    cp_cpu.n_batch = 1;         // Minimal batch for single token
    llama_context* ctx_tgt_cpu = llama_init_from_model(m_tgt_cpu, cp_cpu);

    // EAGLE3: Set draft context type for proper HS isolation
    llama_context_set_draft_context(ctx_dft, true);

    // Get draft-target token mapping
    const int32_t* d2t_map = llama_model_get_eagle_d2t_map(m_dft);

    // 获取内存操作句柄
    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
    llama_memory_t mem_dft = llama_get_memory(ctx_dft);

    // Prompt
    std::string prompt = "The future of artificial intelligence and machine learning is incredibly exciting. We are seeing rapid advances in natural language processing, computer vision, and reinforcement learning.";
    std::vector<llama_token> tokens = common_tokenize(ctx_tgt, prompt, false);
    printf("Prompt length: %zu tokens\n", tokens.size());

    // Warmup run
    printf("Warmup...\n");
    // (这里可以运行一个简化版本，但为了简单，我们直接运行第一次作为warmup)

    // Benchmark runs
    for (int rep = 0; rep < reps; rep++) {
        printf("Run %d/%d...\n", rep + 1, reps);
        fflush(stdout);

        // Reset statistics
        g_stats = {0};
        BenchStats stats;
        uint64_t total_start = get_time_ns();

        // 重置模型状态
        llama_memory_clear(mem_tgt, false);
        llama_memory_clear(mem_dft, false);

        // --- Prompt Phase ---
        uint64_t prompt_start = get_time_ns();
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        llama_decode(ctx_tgt, batch);
        uint64_t prompt_end = get_time_ns();
        stats.prompt_time_ns = prompt_end - prompt_start;

        // --- State Initialization ---
        // 1. Root HS: 来自 Prompt 最后一个 Token
        std::vector<float> current_hs = extract_hs(ctx_tgt, batch.n_tokens - 1);

        // 2. Root Truth: Target 对 Prompt 最后一个 Token 的预测
        llama_token next_ground_truth = get_greedy_token(ctx_tgt, tokens.size() - 1);

        // 记录生成内容
        std::vector<llama_token> generated = tokens;

        // --- Generation Phase ---
        uint64_t gen_start = get_time_ns();

    
    // === MAIN SPECULATIVE LOOP ===
    for (int target_forward_count = 0; target_forward_count < MAX_TARGET_FORWARDS; target_forward_count++) {
        
        // ============================================
        // 1. DRAFT PHASE (Generate K tokens)
        // ============================================
        std::vector<llama_token> draft_tokens;
        std::vector<float> draft_probabilities; // 保存draft token的概率
        llama_token dft_input = next_ground_truth;

        // DEBUG: Print draft phase input information
        // Draft phase (simplified for benchmark)
        
        // Draft loop
        for (int k = 0; k < SPECULATIVE_DEPTH; k++) {

            // Step 0: Use Target HS, Step 1&2: Auto-updated Draft HS
            if (k == 0) {
                llama_context_set_target_hidden_states(ctx_dft, current_hs.data(), current_hs.size());
            } else {
                llama_context_clear_target_hidden_states(ctx_dft);
            }

            // CPU embedding computation for dual-model loading
            llama_context_compute_token_embedding_cpu(ctx_dft, ctx_tgt_cpu, dft_input);
            llama_batch b = llama_batch_get_one(&dft_input, 1);

            int decode_result = llama_decode(ctx_dft, b);
            if (decode_result != 0) {
                break;
            }

            
            const float* logits = llama_get_logits(ctx_dft);
            if (logits == nullptr) {
                break;
            }

            // Greedy Pick
            llama_token best_dft = get_greedy_token(ctx_dft, 0);
            int tgt_id = best_dft + d2t_map[best_dft]; // Map back

            // 计算并保存draft token的概率
            float dft_prob = get_token_probability(ctx_dft, 0, best_dft);

            draft_tokens.push_back(tgt_id);
            draft_probabilities.push_back(dft_prob);
            dft_input = tgt_id;
        }

        
        // Check if no draft tokens were generated
        if (draft_tokens.empty()) {
            break;
        }

        g_stats.total_drafted += draft_tokens.size();
        g_stats.verify_steps++;

        // ============================================
        // 2. VERIFY PHASE (Target Forward Batch)
        // ============================================

        // 【修正 2】构造完整的验证序列：[next_ground_truth] + [draft_tokens]
        // 序列变成: [" the", " problem", " is", " that"]
        std::vector<llama_token> verify_inputs;
        verify_inputs.push_back(next_ground_truth);
        verify_inputs.insert(verify_inputs.end(), draft_tokens.begin(), draft_tokens.end());

        
        // Batch 大小是 verify_inputs 的大小
        llama_batch verify_batch = llama_batch_init(verify_inputs.size(), 0, 1);
        verify_batch.n_tokens = verify_inputs.size();

        // 【修复 BUG 的核心代码】
        int current_pos = generated.size();

        for(int i = 0; i < (int)verify_inputs.size(); i++) {
            verify_batch.token[i] = verify_inputs[i];
            verify_batch.pos[i] = current_pos + i;
            verify_batch.n_seq_id[i] = 1;
            verify_batch.seq_id[i][0] = 0;
            verify_batch.logits[i] = true; // 我们需要每个位置的 Logits
        }

        // Target 真正运行 Forward，更新 KV Cache
        int verify_result = llama_decode(ctx_tgt, verify_batch);
        if (verify_result != 0) {
            break; // 出错时结束循环
        }

        // ============================================
        // 3. COMPARE & UPDATE (Logic)
        // ============================================
        int n_accept = 0;
        llama_token correction_token = -1;
        bool all_accepted = true;

        // 【修正 3】验证逻辑调整
        // verify_inputs: [" the", " problem", " is", " that"]
        // Logits[0] (" the") -> 验证 draft_tokens[0] (" problem")
        // Logits[1] (" problem") -> 验证 draft_tokens[1] (" is")
        // Logits[2] (" is") -> 验证 draft_tokens[2] (" that")

        generated.push_back(next_ground_truth); // 先把基准 token 加入生成序列

        for (int i = 0; i < (int)draft_tokens.size(); i++) {
            llama_token draft_t = draft_tokens[i];

            // 使用 verify_inputs[i] 产生的 Logits 来验证 draft_tokens[i]
            // 注意：verify_inputs[0] 是 " the"，它的输出用于验证 draft_tokens[0] (" problem")
            llama_token truth_t = get_greedy_token(ctx_tgt, i);

            // 【新】概率接收逻辑：使用EAGLE概率接收判断
            float p_target = get_token_probability(ctx_tgt, i, draft_t);
            float p_draft = draft_probabilities[i]; // 需要从draft阶段获取概率

            bool accepted = check_eagle_acceptance(p_target, p_draft, draft_t, truth_t);

            if (accepted) {
                n_accept++;
                if (draft_t != truth_t) {
                    g_stats.prob_accepts++;  // 统计概率性接收
                }
            } else {
                all_accepted = false;
                correction_token = truth_t;
                break;
            }
        }
        
        next_ground_truth = get_greedy_token(ctx_tgt, n_accept); // 更新基准 Token 为下一个位置的预测
        for(int i=0; i<n_accept; i++) generated.push_back(draft_tokens[i]);
        g_stats.total_accepts += n_accept;

        // ============================================
        // 4. ROLLBACK & COMMIT
        // ============================================

        // 更新HS
        if (n_accept >= 1) {
            current_hs = extract_hs(ctx_tgt, 1);
        } else {
            current_hs = extract_hs(ctx_tgt, 0);
        }

        // 2. 更新 tgt model kv cache
        int keep_until = current_pos + 1 + n_accept;
        llama_memory_seq_rm(mem_tgt, 0, keep_until, -1);

        // 3. 统一清理draft cache - 为下一轮做准备
        llama_memory_clear(mem_dft, false);

        // 释放 batch 内存
        llama_batch_free(verify_batch);
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

        printf("\n=== EAGLE3 Benchmark Results (%d Target Forwards) ===\n", MAX_TARGET_FORWARDS);
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
    }

    llama_free(ctx_tgt); llama_free(ctx_tgt_cpu); llama_free(ctx_dft);
    llama_model_free(m_tgt); llama_model_free(m_tgt_cpu); llama_model_free(m_dft);
    llama_backend_free();
    return 0;
}
