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

// ==================== 必要的 API 声明 ====================
// 确保你的 llama.h 中已经声明了这个，或者这里手动声明链接
extern "C" float * llama_context_get_eagle_hs_at_batch_idx(struct llama_context * ctx, int batch_idx, int32_t * out_dim);

// ==================== 配置常量 ====================
const int MAX_TOTAL_TOKENS = 20;      // 总生成长度
const int TOP_K_L1 = 4;               // 第一步 Top K
const int TOP_K_L2 = 4;               // 第二步 Top K (总路径 = 4*4 = 16)
const int TOTAL_LANES = 16;           // 16 个 Lane 刚好对应 16 条路径

// ==================== 全局随机工具 ====================
// 使用固定种子确保结果可重现
static std::mt19937 g_rng(42);  // 固定种子42
static std::uniform_real_distribution<float> g_uniform_dist(0.0f, 1.0f);

// ==================== 辅助结构与函数 ====================

struct DraftPath {
    int lane_id;             // 0 ~ 15
    llama_token t1;          // 第一步预测
    llama_token t2;          // 第二步预测
    float prob_t1;
    float prob_t2;
};

// 统计
struct Stats {
    int total_accepts = 0;
    int total_drafted = 0;
    int verify_steps = 0;
    int prob_accepts = 0;  // 统计概率性接收次数
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
// 遵循公式: r = min(1, P_target / P_draft)
bool check_eagle_acceptance(float p_target, float p_draft, llama_token candidate, llama_token target_greedy) {
    // 1. 如果是 Greedy Match，直接接受 (这是概率接收的特例，p/q 通常 > 1 或最大)
    //    虽然理论上纯随机采样不强制 Greedy，但工程上通常保留 Greedy 路径作为保底。
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

// ==================== 主函数 ====================

int main(int argc, char **argv) {
    setbuf(stdout, NULL); 
    if (argc < 3) { fprintf(stderr, "Usage: %s <tgt> <dft>\n", argv[0]); return 1; }

    common_init();
    llama_backend_init();
    
    // 1. 模型加载
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

    // 3. Prompt 处理 (修复版：Lane 0 跑 -> Copy)
    std::string prompt = "Actually, I think AI";
    std::vector<llama_token> tokens = common_tokenize(ctx_tgt, prompt, true);
    printf("Prompt: %s\n", prompt.c_str());

    // Debug: Print tokens
    printf("[DEBUG] Tokens (%zu): ", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        printf("%d('%s') ", tokens[i], common_token_to_piece(ctx_tgt, tokens[i]).c_str());
    }
    printf("\n");

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
        fprintf(stderr, "Prompt decode failed\n"); return 1;
    }
    llama_batch_free(prompt_batch);

    // 3.2 Sync: Copy Lane 0 to 1..15
    llama_memory_t mem = llama_get_memory(ctx_tgt);
    for (int i = 1; i < TOTAL_LANES; ++i) {
        llama_memory_seq_cp(mem, 0, i, -1, -1);
    }
    printf("[SMP] Prompt synced to all %d lanes.\n", TOTAL_LANES);

    // 4. 初始化状态
    // Prompt Batch 中最后一个 token (Index = tokens.size()-1) 产生了 HS
    std::vector<float> current_hs = extract_hs(ctx_tgt, tokens.size() - 1);
    llama_token next_ground_truth = get_greedy_token(ctx_tgt, tokens.size() - 1);

    if (current_hs.empty()) {
        fprintf(stderr, "[FATAL] Failed to extract initial HS. Check API implementation.\n");
        return 1;
    }
    printf("[SMP] Init GT: %d ('%s')\n", next_ground_truth, common_token_to_piece(ctx_tgt, next_ground_truth).c_str());

    std::vector<llama_token> generated = tokens;

    // ==================== 主循环 ====================
    while (generated.size() < (size_t)(tokens.size() + MAX_TOTAL_TOKENS)) {
        int current_pos = generated.size();
        
        // --- Phase 1: Two-Step Draft ---
        // Step 0: Root -> Top 4 (L1)
        printf("[DEBUG] dft_input = %d ('%s')\n", next_ground_truth, common_token_to_piece(ctx_tgt, next_ground_truth).c_str());
        printf("[DFT INPUT] Step 0: Token %d ('%s') | Using Target HS (%zu dims)\n",
               next_ground_truth, common_token_to_piece(ctx_tgt, next_ground_truth).c_str(), current_hs.size());
        if (!current_hs.empty()) {
            printf("         Layer2[0-2]: %.3f, %.3f, %.3f\n", current_hs[0], current_hs[1], current_hs[2]);
            printf("         Layer18[0-2]: %.3f, %.3f, %.3f\n", current_hs[2560], current_hs[2561], current_hs[2562]);
            printf("         Layer33[0-2]: %.3f, %.3f, %.3f\n", current_hs[5120], current_hs[5121], current_hs[5122]);
        } else {
            printf("         No Target HS available!\n");
        }

        // 清空 Draft 的旧 KV cache
        llama_memory_clear(llama_get_memory(ctx_dft), false);

        llama_context_set_target_hidden_states(ctx_dft, current_hs.data(), current_hs.size());
        llama_batch dft_batch_l1 = llama_batch_get_one(&next_ground_truth, 1);
        llama_decode(ctx_dft, dft_batch_l1);

        auto top_k_l1 = get_top_k_tokens_with_logits(ctx_dft, 0, TOP_K_L1);

        // Step 1: 使用 Step0 的 Top 1 映射到 target vocab 作为输入，进行第二步预测
        // Step 0 的所有 4 个 tokens 共享同一个 draft HS
        llama_token step0_top1_draft = top_k_l1[0].first;  // draft vocab 的 top 1
        llama_token step0_top1_target = step0_top1_draft + d2t_map[step0_top1_draft];  // 映射到 target vocab
        printf("[DEBUG] Step0 Top1: draft=%d ('%s') -> target=%d ('%s')\n",
               step0_top1_draft, common_token_to_piece(ctx_dft, step0_top1_draft).c_str(),
               step0_top1_target, common_token_to_piece(ctx_tgt, step0_top1_target).c_str());

        // Step 1: Top 1 (L1) -> Top 4 (L2)
        printf("[DFT INPUT] Step 1: Token %d ('%s') | Using Auto-updated Draft HS\n",
               step0_top1_target, common_token_to_piece(ctx_tgt, step0_top1_target).c_str());

        // 显示 draft HS 信息（如果有API可用）
        const float* draft_hs = llama_context_get_draft_hidden_states(ctx_dft);
        size_t draft_size = llama_context_get_draft_hidden_states_size(ctx_dft);
        if (draft_hs && draft_size > 0) {
            printf("         Draft HS size: %zu dims\n", draft_size);
            printf("         DraftHS[0-5]: %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n",
                   draft_hs[0], draft_hs[1], draft_hs[2], draft_hs[3], draft_hs[4], draft_hs[5]);
        } else {
            printf("         No draft HS available!\n");
        }

        // 关键：第二步使用 Draft HS，清除 Target HS
        llama_context_clear_target_hidden_states(ctx_dft);
        llama_batch dft_batch_l2 = llama_batch_get_one(&step0_top1_target, 1);
        llama_decode(ctx_dft, dft_batch_l2);

        // 获取 Step 1 的 Top K candidates
        auto top_k_l2 = get_top_k_tokens_with_logits(ctx_dft, 0, TOP_K_L2);

        // 收集所有路径 (4*4 = 16 条路径)
        std::vector<DraftPath> paths;
        int lane_allocator = 0;

        printf("\n[Step %d] Draft Tree (Pos %d):\n", g_stats.verify_steps, current_pos + 1);
        for(int i=0; i<TOP_K_L1; ++i) {
            llama_token t1_dft = top_k_l1[i].first;
            llama_token t1_tgt = t1_dft + d2t_map[t1_dft];

            for(int j=0; j<TOP_K_L2; ++j) {
                llama_token t2_dft = top_k_l2[j].first;
                llama_token t2_tgt = t2_dft + d2t_map[t2_dft];

                paths.push_back({lane_allocator++, t1_tgt, t2_tgt, top_k_l1[i].second, top_k_l2[j].second});
                printf("  Lane %2d: %s -> %s\n",
                    lane_allocator-1,
                    common_token_to_piece(ctx_tgt, t1_tgt).c_str(),
                    common_token_to_piece(ctx_tgt, t2_tgt).c_str());
            }
        }
        g_stats.total_drafted += paths.size() * 2; // Approximated tokens drafted

        // --- Phase 2: Verify Batch (16 Lanes, 3 tokens each) ---
        // 每个 Lane: [Root (P), T1 (P+1), T2 (P+2)]

        int batch_size = paths.size() * 3;
        llama_batch v_batch = llama_batch_init(batch_size, 0, TOTAL_LANES);

        // 记录 Batch Index 用于提取
        // Key: Lane ID. Value: Triple<RootIdx, T1Idx, T2Idx>
        struct LaneIndices { int root; int t1; int t2; };
        std::map<int, LaneIndices> lane_indices;

        for(const auto& path : paths) {
            int l = path.lane_id;

            // Root (P)
            add_lane_token(v_batch, next_ground_truth, current_pos, l, true); // 需要 Root Logits
            int idx_root = v_batch.n_tokens - 1;

            // T1 (P+1)
            add_lane_token(v_batch, path.t1, current_pos + 1, l, true); // 需要 T1 Logits
            int idx_t1 = v_batch.n_tokens - 1;

            // T2 (P+2)
            add_lane_token(v_batch, path.t2, current_pos + 2, l, true);
            int idx_t2 = v_batch.n_tokens - 1;

            lane_indices[l] = {idx_root, idx_t1, idx_t2};
        }

        if (llama_decode(ctx_tgt, v_batch) != 0) { printf("Decode failed\n"); break; }

        // --- Phase 3: Elect Winner (Probabilistic Verification) ---

        // 3.1 验证 Level 1
        // 共享 Root Logits，随便取一个 lane (e.g., lane 0) 的 root logits
        int root_logits_idx = lane_indices[0].root;
        llama_token root_greedy_prediction = get_greedy_token(ctx_tgt, root_logits_idx);

        int accepted_l1_idx = -1; // 在 top_k_l1 中的索引
        llama_token accepted_l1_token = 0;
        int best_lane = -1; // 将用于同步的 lane
        int max_len = 0;

        printf("\n[Verify Step %d]\n", g_stats.verify_steps);

        // 遍历 L1 候选者 (按 Draft 概率从高到低)
        for(int i = 0; i < TOP_K_L1; ++i) {
            llama_token candidate = top_k_l1[i].first + d2t_map[top_k_l1[i].first];
            float p_draft = top_k_l1[i].second;
            float p_target = get_token_probability(ctx_tgt, root_logits_idx, candidate);

            bool accepted = check_eagle_acceptance(p_target, p_draft, candidate, root_greedy_prediction);

            if (accepted) {
                printf("  L1 Accept: %s (Pt=%.3f, Pd=%.3f)\n", common_token_to_piece(ctx_tgt, candidate).c_str(), p_target, p_draft);
                if (candidate != root_greedy_prediction) g_stats.prob_accepts++;

                accepted_l1_idx = i;
                accepted_l1_token = candidate;
                break; // 找到第一个接受的即停止 (EAGLE 策略之一)
            } else {
                // printf("  L1 Reject: %s (Pt=%.3f, Pd=%.3f)\n", common_token_to_piece(ctx_tgt, candidate).c_str(), p_target, p_draft);
            }
        }

        // 3.2 验证 Level 2 (仅当 L1 被接受且是展开分支时)
        if (accepted_l1_idx != -1) {
            // L1 验证通过。
            // 检查我们是否有权验证 L2：Draft 模型只计算了 top_k_l1[0] 的后续。
            // 如果 accepted_l1_idx != 0，说明 Draft 预测的 L2 都是基于错误的父节点的，无效。

            if (accepted_l1_idx == 0) {
                // 我们可以验证 L2
                // 需要找到一个 path，它的 t1 == accepted_l1_token。
                // Lane 0 肯定满足 (因为生成逻辑)，且包含 T1 的 logits。
                int t1_logits_idx = lane_indices[0].t1;
                llama_token t1_greedy_prediction = get_greedy_token(ctx_tgt, t1_logits_idx);

                int accepted_l2_idx = -1;
                llama_token accepted_l2_token = 0;

                for(int j = 0; j < TOP_K_L2; ++j) {
                    llama_token candidate = top_k_l2[j].first + d2t_map[top_k_l2[j].first];
                    float p_draft = top_k_l2[j].second;
                    // 注意：这里的 Target Prob 是基于 accepted_l1_token 的上下文计算的
                    float p_target = get_token_probability(ctx_tgt, t1_logits_idx, candidate);

                    bool accepted = check_eagle_acceptance(p_target, p_draft, candidate, t1_greedy_prediction);

                    if (accepted) {
                        printf("  L2 Accept: %s (Pt=%.3f, Pd=%.3f)\n", common_token_to_piece(ctx_tgt, candidate).c_str(), p_target, p_draft);
                        if (candidate != t1_greedy_prediction) g_stats.prob_accepts++;

                        accepted_l2_idx = j;
                        accepted_l2_token = candidate;
                        break;
                    }
                }

                if (accepted_l2_idx != -1) {
                    // Hit 2
                    max_len = 2;
                    // 寻找对应的 lane 用于同步
                    for(const auto& p : paths) {
                        if(p.t1 == accepted_l1_token && p.t2 == accepted_l2_token) {
                            best_lane = p.lane_id; break;
                        }
                    }
                } else {
                    // Hit 1 (L2 rejected)
                    max_len = 1;
                    // 寻找任意包含正确 L1 的 lane
                    for(const auto& p : paths) {
                        if(p.t1 == accepted_l1_token) {
                            best_lane = p.lane_id; break;
                        }
                    }
                }

            } else {
                // L1 接受了，但不是 Draft 展开的那条路 (accepted_l1_idx != 0)。
                // 无法利用 draft 的 L2 预测。
                max_len = 1;
                // 寻找任意包含正确 L1 的 lane (path loop 生成了所有组合，所以肯定有)
                for(const auto& p : paths) {
                    if(p.t1 == accepted_l1_token) {
                        best_lane = p.lane_id; break;
                    }
                }
                printf("  L1 Accepted (Non-Top1), skipping L2 verify.\n");
            }
        } else {
            // Miss (L1 Rejected)
            max_len = 0;
        }

        // --- Phase 4: Merge & Sync ---
        generated.push_back(next_ground_truth); // Root Always Accepted

        if (max_len == 2) {
            // Hit 2 Tokens (Root + T1 + T2)
            auto idx = lane_indices[best_lane];
            // 使用已接受的 token，而不是重新计算 greedy token
            llama_token t1 = accepted_l1_token;
            llama_token t2 = 0;
            // 从 path 中找到对应的 t2
            for(const auto& p : paths) {
                if(p.lane_id == best_lane && p.t1 == accepted_l1_token) {
                    t2 = p.t2; break;
                }
            }

            printf("  -> HIT 2! Lane %d (%s -> %s)\n", best_lane, common_token_to_piece(ctx_tgt, t1).c_str(), common_token_to_piece(ctx_tgt, t2).c_str());
            g_stats.total_accepts += 2;
            generated.push_back(t1);
            generated.push_back(t2);

            // Extract from T2 pos (Pos+2)
            current_hs = extract_hs(ctx_tgt, idx.t2);
            next_ground_truth = get_greedy_token(ctx_tgt, idx.t2);

            // Sync [P, P+3) - 全部清除然后完整复制
            llama_memory_t mem_sync = llama_get_memory(ctx_tgt);
            for(int i=0; i<TOTAL_LANES; ++i) {
                if(i != best_lane) {
                    llama_memory_seq_rm(mem_sync, i, -1, -1);
                    llama_memory_seq_cp(mem_sync, best_lane, i, -1, -1);
                }
            }

        } else if (max_len == 1) {
            // Hit 1 Token (Root + T1)
            auto idx = lane_indices[best_lane];
            // 使用已接受的 L1 token
            llama_token t1 = accepted_l1_token;

            printf("  -> HIT 1! Lane %d (%s)\n", best_lane, common_token_to_piece(ctx_tgt, t1).c_str());
            g_stats.total_accepts += 1;
            generated.push_back(t1);

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
            // Miss (Root Only)
            auto idx = lane_indices[0]; // Any lane works for Root
            printf("  -> MISS. Correcting to %s\n", common_token_to_piece(ctx_tgt, root_greedy_prediction).c_str());

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
        // 根据 max_len 动态清理
        // Hit 2: Keep P, P+1, P+2. Rm P+3.
        // Hit 1: Keep P, P+1. Rm P+2.
        // Miss: Keep P. Rm P+1.
        int keep_len = (max_len == 2) ? 3 : (max_len == 1 ? 2 : 1);
        llama_memory_t mem = llama_get_memory(ctx_tgt);
        llama_memory_seq_rm(mem, -1, current_pos + keep_len, -1);
        
        g_stats.verify_steps++;
        llama_batch_free(v_batch);

        // Stream Output
        printf("Stream: ");
        for(size_t i=tokens.size(); i<generated.size(); i++) 
            printf("%s", common_token_to_piece(ctx_tgt, generated[i]).c_str());
        printf("\n");
        fflush(stdout);
    }

    // ==================== 统计与退出 ====================
    printf("\n=== Statistics ===\n");
    printf("Total Drafted: %d\n", g_stats.total_drafted);
    printf("Total Accepted: %d\n", g_stats.total_accepts);
    printf("Probabilistic Accepts: %d\n", g_stats.prob_accepts);
    printf("Acceptance Rate: %.2f%%\n", 100.0f * g_stats.total_accepts / g_stats.total_drafted);
    printf("Mean Speculation Length: %.2f\n", (float)g_stats.total_accepts / g_stats.verify_steps);

    llama_free(ctx_tgt); llama_free(ctx_dft);
    llama_model_free(m_tgt); llama_model_free(m_dft);
    llama_backend_free();
    return 0;
}
