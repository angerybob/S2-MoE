#include "common.h"
#include "llama.h"
#include <vector>
#include <cstdio>
#include <algorithm>
#include <string>
#include <cmath>
#include <random>
#include <map>

// Forward declarations
llama_token get_greedy_token(llama_context* ctx, int batch_idx);
std::vector<float> extract_hs(llama_context* ctx, int batch_idx);

// === 配置区域 ===
const int MAX_TOTAL_TOKENS = 50;      // 总生成长度
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

    // **新增**：数据有效性检查
    bool data_suspicious = true;
    bool has_nan_inf = false;
    int non_zero_count = 0;
    float min_val = res[0], max_val = res[0];

    for (int i = 0; i < std::min(50, dims); i++) {  // 检查前50个元素
        if (std::isnan(res[i]) || std::isinf(res[i])) {
            has_nan_inf = true;
            break;
        }
        if (res[i] != 0.0f) {
            data_suspicious = false;
            non_zero_count++;
        }
        min_val = std::min(min_val, res[i]);
        max_val = std::max(max_val, res[i]);
    }

    if (has_nan_inf) {
        printf("[警告] 提取的隐藏状态包含NaN或Inf值！\n");
    } else if (data_suspicious) {
        printf("[警告] 提取的隐藏状态数据可疑（前50个元素中几乎全为零）\n");
        printf("         非零元素: %d/50, 最小值: %.6f, 最大值: %.6f\n",
               non_zero_count, min_val, max_val);
    } else {
        printf("[信息] 隐藏状态数据验证通过，非零元素: %d/50\n", non_zero_count);
    }

    printf("[DEBUG] Extracted HS for batch_idx %d: %d dims\n", batch_idx, dims);
    printf("         数据范围: [%.6f, %.6f]\n", min_val, max_val);
    printf("         Layer2[0-2]: %.3f, %.3f, %.3f\n", res[0], res[1], res[2]);
    printf("         Layer18[0-2]: %.3f, %.3f, %.3f\n", res[2560], res[2561], res[2562]);
    printf("         Layer33[0-2]: %.3f, %.3f, %.3f\n", res[5120], res[5121], res[5122]);

    return res;
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL); 

    if (argc < 3) { fprintf(stderr, "Usage: %s <tgt> <dft>\n", argv[0]); return 1; }

    common_init();
    llama_backend_init();
    
    // Load Models
    llama_model_params mp = llama_model_default_params();
    llama_model* m_tgt = llama_model_load_from_file(argv[1], mp);
    llama_model* m_dft = llama_model_load_from_file(argv[2], mp);
    
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; cp.n_batch = 512; // Batch必须足够大以容纳 verification batch
    
    cp.embeddings = true; // Target need HS (pre-attention hidden states for EAGLE3)
    llama_context* ctx_tgt = llama_init_from_model(m_tgt, cp);
    cp.embeddings = false;
    printf("[DEBUG] About to initialize ctx_dft...\n");
    llama_context* ctx_dft = llama_init_from_model(m_dft, cp);
    printf("[DEBUG] ctx_dft initialized: %p\n", (void*)ctx_dft);
    if (!ctx_dft) {
        printf("[ERROR] Failed to initialize draft context!\n");
        return -1;
    }
    printf("[DEBUG] ctx_dft initialization completed successfully\n");

    // NEW: Create CPU target model and context for embeddings only (Phase 1)
    llama_model_params mp_cpu = llama_model_default_params();
    mp_cpu.n_gpu_layers = 0;     // Force CPU-only loading
    llama_model* m_tgt_cpu = llama_model_load_from_file(argv[1], mp_cpu);

    // Verify CPU model loaded successfully
    printf("[DEBUG] CPU target model loaded: m_tgt_cpu=%p\n", (void*)m_tgt_cpu);

    llama_context_params cp_cpu = llama_context_default_params();
    cp_cpu.n_ctx = 2048;
    cp_cpu.embeddings = true;    // Need embeddings
    cp_cpu.n_batch = 1;         // Minimal batch for single token
    llama_context* ctx_tgt_cpu = llama_init_from_model(m_tgt_cpu, cp_cpu);

    // Verify CPU context created successfully and check embedding tensor
    printf("[DEBUG] CPU target context created: ctx_tgt_cpu=%p\n", (void*)ctx_tgt_cpu);
    if (ctx_tgt_cpu) {
        const llama_model* model_cpu = llama_get_model(ctx_tgt_cpu);
        if (model_cpu) {
            const struct llama_vocab * vocab = llama_model_get_vocab(model_cpu);
            printf("[DEBUG] CPU model loaded successfully, vocab_size=%d\n",
                   llama_vocab_n_tokens(vocab));
        }
    }

    // EAGLE3: Set draft context type for proper HS isolation
    llama_context_set_draft_context(ctx_dft, true);

    // Get draft-target token mapping
    const int32_t* d2t_map = llama_model_get_eagle_d2t_map(m_dft);

    // 获取内存操作句柄 (步骤A)
    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
    printf("[DEBUG] About to get memory for ctx_dft: %p\n", (void*)ctx_dft);
    llama_memory_t mem_dft = llama_get_memory(ctx_dft);
    printf("Memory handles obtained: tgt=%p, dft=%p\n", (void*)mem_tgt, (void*)mem_dft);

    // Prompt Prefill
    std::string prompt = "Actually, I think AI";
    std::vector<llama_token> tokens = common_tokenize(ctx_tgt, prompt, false);
    printf("Prompt: %s\n", prompt.c_str());

    // Target Run Prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_decode(ctx_tgt, batch);

    // --- State Initialization ---
    // 1. Root HS: 来自 Prompt 最后一个 Token (当前batch的最后一个token)
    printf("[Prompt Decode] Extracting HS from batch size %d, last token index %d\n",
           batch.n_tokens, batch.n_tokens - 1);
    std::vector<float> current_hs = extract_hs(ctx_tgt, batch.n_tokens - 1);
    
    // 2. Root Truth: Target 对 Prompt 最后一个 Token 的预测 (用于验证 Draft 第 1 步)
    llama_token next_ground_truth = get_greedy_token(ctx_tgt, tokens.size() - 1);
    
    
    // 记录生成内容
    std::vector<llama_token> generated = tokens;

    
    // === MAIN SPECULATIVE LOOP ===
    while (generated.size() < (size_t)(tokens.size() + MAX_TOTAL_TOKENS)) {
        // 检查是否已经达到最大生成长度
        if (generated.size() - tokens.size() >= (size_t)MAX_TOTAL_TOKENS) {
            break;
        }
        
        // ============================================
        // 1. DRAFT PHASE (Generate K tokens)
        // ============================================
        std::vector<llama_token> draft_tokens;
        std::vector<float> draft_probabilities; // 保存draft token的概率
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
        for (int k = 0; k < SPECULATIVE_DEPTH; k++) {

            // Step 0: Use Target HS (7680 dims), Step 1&2: Auto-updated Draft HS (2560 dims)
            if (k == 0) {
                llama_context_set_target_hidden_states(ctx_dft, current_hs.data(), current_hs.size());
            } else {
                // EAGLE3: Step 1&2 use draft HS, so clear target HS to prevent it from being used
                llama_context_clear_target_hidden_states(ctx_dft);
            }

            // CPU embedding computation for dual-model loading
            llama_context_compute_token_embedding_cpu(ctx_dft, ctx_tgt_cpu, dft_input);

            llama_batch b = llama_batch_get_one(&dft_input, 1);

            // 输出dft model实际使用的HS输入
            if (k == 0) {
                // Step 0: 使用Target HS (7680维)
                printf("[DFT INPUT] Step %d: Token %d ('%s') | Using Target HS (7680 dims)\n",
                       k, dft_input, common_token_to_piece(ctx_tgt, dft_input).c_str());
                printf("         Layer2[0-2]: %.3f, %.3f, %.3f\n", current_hs[0], current_hs[1], current_hs[2]);
                printf("         Layer18[0-2]: %.3f, %.3f, %.3f\n", current_hs[2560], current_hs[2561], current_hs[2562]);
                printf("         Layer33[0-2]: %.3f, %.3f, %.3f\n", current_hs[5120], current_hs[5121], current_hs[5122]);
            } else {
                // Step 1+: 将使用auto-updated Draft HS (2560维，来自上一次decode)
                // 使用专门的draft HS API，而不是EAGLE HS API
                const float* draft_hs = llama_context_get_draft_hidden_states(ctx_dft);
                size_t draft_size = llama_context_get_draft_hidden_states_size(ctx_dft);

                printf("[DFT INPUT] Step %d: Token %d ('%s') | Using Auto-updated Draft HS (%zu dims):\n",
                       k, dft_input, common_token_to_piece(ctx_tgt, dft_input).c_str(), draft_size);
                if (draft_hs && draft_size > 0) {
                    printf("         DraftHS[0-5]: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n", draft_hs[0], draft_hs[1], draft_hs[2], draft_hs[3], draft_hs[4], draft_hs[5]);
                } else {
                    printf("         No draft HS available!\n");
                }
            }

            int decode_result = llama_decode(ctx_dft, b);
            if (decode_result != 0) {
                break;
            }

            // 输出decode后auto-updated的draft HS状态（用于调试）
            // 注意：auto_update_draft_hidden_states()在llama_decode()后自动调用
            const float* updated_draft_hs = llama_context_get_draft_hidden_states(ctx_dft);
            size_t updated_draft_size = llama_context_get_draft_hidden_states_size(ctx_dft);

            if (updated_draft_hs && updated_draft_size > 0) {
                printf("[DFT AUTO-UPDATED HS] Step %d: %zu dims (after auto_update)\n", k, updated_draft_size);
                printf("         DraftHS[0-5]: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
                       updated_draft_hs[0], updated_draft_hs[1], updated_draft_hs[2],
                       updated_draft_hs[3], updated_draft_hs[4], updated_draft_hs[5]);
            } else {
                printf("[DFT AUTO-UPDATED HS] Step %d: No draft HS available after auto_update\n", k);
            }

            const float* logits = llama_get_logits(ctx_dft);
            if (logits == nullptr) {
                break;
            }

            // Greedy Pick
            llama_token best_dft = get_greedy_token(ctx_dft, 0);
            int tgt_id = best_dft + d2t_map[best_dft]; // Map back

            // 【新】计算并保存draft token的概率
            float dft_prob = get_token_probability(ctx_dft, 0, best_dft);

            printf("Draft prediction: DFT=%d ('%s') -> TGT=%d ('%s') [P=%.3f]\n",
                   best_dft, common_token_to_piece(ctx_dft, best_dft).c_str(),
                   tgt_id, common_token_to_piece(ctx_tgt, tgt_id).c_str(), dft_prob);

            draft_tokens.push_back(tgt_id);
            draft_probabilities.push_back(dft_prob);
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
                    printf("  [Prob Accept] %s: Pt=%.3f, Pd=%.3f, r=%.3f\n",
                           common_token_to_piece(ctx_tgt, draft_t).c_str(), p_target, p_draft, p_target/p_draft);
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

        // 1. 统一更新HS - 提取正确的token的HS
        // 关键理解：
        // - Root token: verify_inputs[0] - 这是ground truth，必须接受
        // - Draft prediction: verify_inputs[1] - 可能被接受或拒绝
        // - EAGLE逻辑：提取被接受的最后一个token的HS用于下一轮draft

        if (n_accept >= 1) {
            // HIT: Draft prediction被接受，提取draft prediction的HS (batch_idx=1)
            // 这是被接受的最后一个token
            printf("[HS Update] HIT: Extracting HS for draft prediction (batch_idx=1, n_accept=%d)\n", n_accept);

            current_hs = extract_hs(ctx_tgt, 1);

            printf("[DEBUG] Extracted HS for accepted draft token %d ('%s')\n",
                   verify_inputs[1],
                   common_token_to_piece(ctx_tgt, verify_inputs[1]).c_str());
        } else {
            // MISS: Draft prediction被拒绝，只接受root token，提取root token的HS (batch_idx=0)
            printf("[HS Update] MISS: Extracting HS for root token (batch_idx=0)\n");

            current_hs = extract_hs(ctx_tgt, 0);

            printf("[DEBUG] Extracted HS for root token %d ('%s')\n",
                   verify_inputs[0],
                   common_token_to_piece(ctx_tgt, verify_inputs[0]).c_str());
        }

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
    printf("Probabilistic Accepts: %d\n", g_stats.prob_accepts);
    printf("Verification Steps: %d\n", g_stats.verify_steps);
    printf("Acceptance Rate: %.2f%%\n", 100.0f * g_stats.total_accepts / g_stats.total_drafted);
    printf("Mean Speculation Length: %.2f\n", (float)g_stats.total_accepts / g_stats.verify_steps);

    llama_free(ctx_tgt); llama_free(ctx_tgt_cpu); llama_free(ctx_dft);
    llama_model_free(m_tgt); llama_model_free(m_tgt_cpu); llama_model_free(m_dft);
    llama_backend_free();
    return 0;
}
