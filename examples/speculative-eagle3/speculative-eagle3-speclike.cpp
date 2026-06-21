#include "arg.h"
#include "common.h"
#include "llama.h"
#include <vector>
#include <cstdio>
#include <algorithm>
#include <string>
#include <cmath>
#include <random>
#include <chrono>

// Forward declarations
llama_token get_greedy_token(llama_context* ctx, int batch_idx);
std::vector<float> extract_hs(llama_context* ctx, int batch_idx);
float get_token_probability(llama_context* ctx, int batch_idx, llama_token token);
float generate_uniform_random();

const std::vector<int> EAGLE_LAYERS = {2, 18, 33}; 

// 统计
struct Stats {
    int total_accepts = 0;
    int total_drafted = 0;
    int verify_steps = 0;
    int strict_rejections = 0;        // 非最优token被拒绝次数
    int probabilistic_accepts = 0;   // 概率性接受次数
} g_stats;

// 全局随机数生成器
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> uniform_dist(0.0f, 1.0f);

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

// 辅助：从 Batch 特定位置提取 HS (SGLang-compatible: pre-attention hidden states)
// 注意：现在提取的是 layer 的输入 (pre-attention)，而不是 layer 的输出
std::vector<float> extract_hs(llama_context* ctx, int batch_idx) {
    std::vector<float> res;
    const int32_t fallback_dim = llama_model_n_embd(llama_get_model(ctx));
    for (int layer : EAGLE_LAYERS) {
        int32_t dim = 0;
        float* ptr = llama_context_extract_hidden_states_from_layer(ctx, batch_idx, layer, &dim);

        if (ptr) {
            // 检查维度是否合理
            if (dim <= 0 || dim > 100000) {  // 合理的维度范围检查
                free(ptr);
                return {};
            }

            res.insert(res.end(), ptr, ptr + dim);
            free(ptr);
        } else {
            return std::vector<float>(3 * fallback_dim, 0.0f);
        }
    }
    return res;
}

// 辅助：获取特定token的概率
float get_token_probability(llama_context* ctx, int batch_idx, llama_token token) {
    auto* logits = llama_get_logits(ctx);
    const auto* vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n_vocab = llama_vocab_n_tokens(vocab);
    const float* current_logits = logits + (batch_idx * n_vocab);

    // 找到最大logit用于数值稳定性
    float max_logit = current_logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (current_logits[i] > max_logit) {
            max_logit = current_logits[i];
        }
    }

    // 计算softmax概率
    float sum_exp = 0.0f;
    float token_exp = 0.0f;
    for (int i = 0; i < n_vocab; i++) {
        float exp_val = expf(current_logits[i] - max_logit);
        sum_exp += exp_val;
        if (i == token) {
            token_exp = exp_val;
        }
    }

    return token_exp / sum_exp;
}

// 辅助：生成均匀随机数
float generate_uniform_random() {
    return uniform_dist(gen);
}

// 新的验证逻辑：仿照Speculative Decoding的Rejection Sampling
bool verify_token_with_rejection_sampling(llama_context* ctx_tgt, int batch_idx,
                                        llama_token draft_token,
                                        float draft_prob) {
    // 获取Target模型对draft token的概率
    float target_prob = get_token_probability(ctx_tgt, batch_idx, draft_token);

    // 获取Target模型的最优token
    llama_token target_best = get_greedy_token(ctx_tgt, batch_idx);

    printf("[VERIFY] Draft token: %d ('%s')\n",
           draft_token, common_token_to_piece(ctx_tgt, draft_token).c_str());
    printf("[VERIFY] Target best: %d ('%s')\n",
           target_best, common_token_to_piece(ctx_tgt, target_best).c_str());
    printf("[VERIFY] Target probability: %.6f\n", target_prob);

    // 检查是否为最优token（Greedy模式）
    if (draft_token == target_best) {
        printf("[VERIFY] ✅ Greedy match - ACCEPTED\n");
        return true;
    }

    // 否则使用Rejection Sampling（概率模式）- 标准公式
    float acceptance_prob = std::min(1.0f, target_prob / draft_prob);
    float uniform_random = generate_uniform_random();

    printf("[VERIFY] Draft probability: %.6f\n", draft_prob);
    printf("[VERIFY] Acceptance probability: %.6f\n", acceptance_prob);
    printf("[VERIFY] Uniform random: %.6f\n", uniform_random);

    if (uniform_random <= acceptance_prob && draft_prob > 0.0f) {
        printf("[VERIFY] ✅ Probabilistic acceptance - ACCEPTED\n");
        g_stats.probabilistic_accepts++;
        return true;
    } else {
        printf("[VERIFY] ❌ Rejected - will use target prediction\n");
        g_stats.strict_rejections++;
        return false;
    }
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL); 

    common_params params;
    if (argc == 3 && argv[1][0] != '-' && argv[2][0] != '-') {
        params.model.path = argv[1];
        params.speculative.model.path = argv[2];
        params.prompt = "I can help you with that.";
        params.n_predict = 20;
        params.n_ctx = 2048;
        params.n_batch = 512;
        params.n_ubatch = 512;
        params.speculative.n_max = 3;
    } else {
        if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
            return 1;
        }
    }

    if (params.model.path.empty() || params.speculative.model.path.empty()) {
        fprintf(stderr, "Usage: %s -m <target> -md <eagle3-draft> [common speculative args]\n", argv[0]);
        return 1;
    }
    if (params.prompt.empty()) {
        params.prompt = "I can help you with that.";
    }
    const int max_total_tokens = params.n_predict < 0 ? 20 : params.n_predict;
    const int speculative_depth = std::max(1, params.speculative.n_max);

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);
    
    // Load Models
    llama_model_params mp_tgt = common_model_params_to_llama(params);
    llama_model* m_tgt = llama_model_load_from_file(params.model.path.c_str(), mp_tgt);
    if (m_tgt == nullptr) {
        fprintf(stderr, "failed to load target model: %s\n", params.model.path.c_str());
        return 1;
    }

    common_params params_dft = params;
    params_dft.devices = params.speculative.devices;
    params_dft.model = params.speculative.model;
    params_dft.n_gpu_layers = params.speculative.n_gpu_layers;
    params_dft.n_expert_used = params.speculative.n_expert_used;
    if (params.speculative.n_ctx > 0) {
        params_dft.n_ctx = params.speculative.n_ctx;
    }
    if (params.speculative.cpuparams.n_threads > 0) {
        params_dft.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }
    params_dft.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    params_dft.tensor_buft_overrides = params.speculative.tensor_buft_overrides;

    llama_model_params mp_dft = common_model_params_to_llama(params_dft);
    llama_model* m_dft = llama_model_load_from_file(params_dft.model.path.c_str(), mp_dft);
    if (m_dft == nullptr) {
        fprintf(stderr, "failed to load draft model: %s\n", params_dft.model.path.c_str());
        llama_model_free(m_tgt);
        return 1;
    }

    llama_context_params cp_tgt = common_context_params_to_llama(params);
    llama_context_params cp_dft = common_context_params_to_llama(params_dft);
    
    cp_tgt.embeddings = true; // Target need HS (pre-attention hidden states for EAGLE3)
    llama_context* ctx_tgt = llama_init_from_model(m_tgt, cp_tgt);
    cp_dft.embeddings = false;
    llama_context* ctx_dft = llama_init_from_model(m_dft, cp_dft);
    if (ctx_tgt == nullptr || ctx_dft == nullptr) {
        fprintf(stderr, "failed to create EAGLE3 contexts\n");
        if (ctx_tgt != nullptr) llama_free(ctx_tgt);
        if (ctx_dft != nullptr) llama_free(ctx_dft);
        llama_model_free(m_tgt);
        llama_model_free(m_dft);
        return 1;
    }

    // EAGLE3: Set draft context type for proper HS isolation
    llama_context_set_draft_context(ctx_dft, true);

    // Bind
    llama_context_set_target_embedding_layer(ctx_dft, ctx_tgt);
    // 获取内存操作句柄 (步骤A)
    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
    llama_memory_t mem_dft = llama_get_memory(ctx_dft);
    // printf("Memory handles obtained: tgt=%p, dft=%p\n", (void*)mem_tgt, (void*)mem_dft);

    // Prompt Prefill
    std::vector<llama_token> tokens = common_tokenize(ctx_tgt, params.prompt, true, true);
    printf("Prompt: %s\n", params.prompt.c_str());

    // Target Run Prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_decode(ctx_tgt, batch);

    // --- State Initialization ---
    // 1. Root HS: 来自 Prompt 最后一个 Token
    std::vector<float> current_hs = extract_hs(ctx_tgt, tokens.size() - 1);
    
    // 2. Root Truth: Target 对 Prompt 最后一个 Token 的预测 (用于验证 Draft 第 1 步)
    llama_token next_ground_truth = get_greedy_token(ctx_tgt, tokens.size() - 1);
    
    
    // 记录生成内容
    std::vector<llama_token> generated = tokens;

    
    // === MAIN SPECULATIVE LOOP ===
    while (generated.size() < (size_t)(tokens.size() + max_total_tokens)) {
        // 检查是否已经达到最大生成长度
        if (generated.size() - tokens.size() >= (size_t)max_total_tokens) {
            break;
        }
        
        // ============================================
        // 1. DRAFT PHASE (Generate K tokens)
        // ============================================
        std::vector<llama_token> draft_tokens;
        std::vector<float> draft_probs;  // 保存draft概率
        llama_token dft_input = next_ground_truth;

        // DEBUG: Print draft phase input information
        printf("[DEBUG] DRAFT PHASE START\n");
        printf("[DEBUG] dft_input = %d ('%s')\n", dft_input, common_token_to_piece(ctx_tgt, dft_input).c_str());
        printf("[DEBUG] current sequence: [");
        for(size_t i=0; i<generated.size(); i++) {
            printf("%d('%s')", generated[i], common_token_to_piece(ctx_tgt, generated[i]).c_str());
            if(i < generated.size()-1) printf(", ");
        }
        printf("]\n");
        
        // Draft loop
        for (int k = 0; k < speculative_depth; k++) {

            // Step 0: Use Target HS (7680 dims), Step 1&2: Auto-updated Draft HS (2560 dims)
            if (k == 0) {
                llama_context_set_target_hidden_states(ctx_dft, current_hs.data(), current_hs.size());
            } else {
                // EAGLE3: Step 1&2 use draft HS, so clear target HS to prevent it from being used
                llama_context_clear_target_hidden_states(ctx_dft);
            }

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
            llama_token tgt_id = get_greedy_token(ctx_dft, 0);

            // 保存draft token的真实概率
            float draft_prob = get_token_probability(ctx_dft, 0, tgt_id);

            printf("Draft prediction: TGT=%d ('%s'), prob=%.6f\n",
                   tgt_id, common_token_to_piece(ctx_tgt, tgt_id).c_str(),
                   draft_prob);

            draft_tokens.push_back(tgt_id);
            draft_probs.push_back(draft_prob);
            dft_input = tgt_id;
        }

        printf("Draft chain: ");
        for(auto t : draft_tokens) printf("'%s' ", common_token_to_piece(ctx_tgt, t).c_str());
        printf("\n");

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

        // DEBUG: Print verify phase input information
        printf("[DEBUG] VERIFY PHASE START\n");
        printf("[DEBUG] verify_inputs = [");
        for(size_t i=0; i<verify_inputs.size(); i++) {
            printf("%d('%s')", verify_inputs[i], common_token_to_piece(ctx_tgt, verify_inputs[i]).c_str());
            if(i < verify_inputs.size()-1) printf(", ");
        }
        printf("] (size=%zu)\n", verify_inputs.size());

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
            printf("\n--- Verifying draft token %d/%d ---\n", i + 1, (int)draft_tokens.size());

            // 使用Rejection Sampling验证
            bool is_accepted = verify_token_with_rejection_sampling(ctx_tgt, i, draft_tokens[i], draft_probs[i]);

            if (is_accepted) {
                n_accept++;
            } else {
                all_accepted = false;
                correction_token = get_greedy_token(ctx_tgt, i);
                break;
            }
        }
        
        next_ground_truth = get_greedy_token(ctx_tgt, n_accept); // 更新基准 Token 为下一个位置的预测
        for(int i=0; i<n_accept; i++) generated.push_back(draft_tokens[i]);
        g_stats.total_accepts += n_accept;

        // ============================================
        // 4. ROLLBACK & COMMIT
        // ============================================

        // 1. 统一更新HS - 提取最后被接受的token的HS
        // verify_inputs = [next_ground_truth] + [draft_tokens]
        // 最后被接受的token的位置就是 n_accept
        printf("[HS Update] Extracting HS from position %d (last accepted token)\n", n_accept);
        current_hs = extract_hs(ctx_tgt, n_accept);

        // 2. 更新 tgt model kv cache
        int keep_until = current_pos + 1 + n_accept;
        llama_memory_seq_rm(mem_tgt, 0, keep_until, -1);

        // 3. 统一清理draft cache - 为下一轮做准备
        llama_memory_clear(mem_dft, false);

        // 释放 batch 内存
        llama_batch_free(verify_batch);

        // 打印当前 Stream
        printf("Stream: ");
        for(size_t i=tokens.size(); i<generated.size(); i++) {
            printf("%s", common_token_to_piece(ctx_tgt, generated[i]).c_str());
        }
        printf("\n");
        fflush(stdout);
    }

    printf("\n=== Statistics ===\n");
    printf("Total Drafted: %d\n", g_stats.total_drafted);
    printf("Total Accepted: %d\n", g_stats.total_accepts);
    printf("Acceptance Rate: %.2f%%\n", g_stats.total_drafted > 0 ? 100.0f * g_stats.total_accepts / g_stats.total_drafted : 0.0f);
    printf("Verification Steps: %d\n", g_stats.verify_steps);
    printf("Mean Speculation Length: %.2f\n", g_stats.verify_steps > 0 ? (float)g_stats.total_accepts / g_stats.verify_steps : 0.0f);
    printf("Probabilistic Accepts: %d\n", g_stats.probabilistic_accepts);
    printf("Strict Rejections: %d\n", g_stats.strict_rejections);

    llama_free(ctx_tgt); llama_free(ctx_dft);
    llama_model_free(m_tgt); llama_model_free(m_dft);
    llama_backend_free();
    return 0;
}
