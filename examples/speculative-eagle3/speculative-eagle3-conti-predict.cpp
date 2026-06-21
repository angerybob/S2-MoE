#include "common.h"
#include "llama.h"
#include <vector>
#include <cstdio>
#include <algorithm>
#include <string>
#include <cmath>

// === 配置区域 ===
const int MAX_STEPS = 20;
const std::vector<int> EAGLE_LAYERS = {2, 18, 33}; // 需与 config.json 一致
const int TOP_K_SHOW = 5; // 打印Draft前几名

struct StepResult {
    llama_token token_id;
    std::string token_str;
    float logit;
};

// Forward declaration
StepResult sample_greedy(llama_context* ctx);

// 辅助：获取Top1 Token
StepResult sample_greedy(llama_context* ctx) {
    auto* logits = llama_get_logits(ctx);
    auto* model = llama_get_model(ctx);
    const auto* vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    int best_id = 0;
    float max_val = -1e9;
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            best_id = i;
        }
    }
    return {best_id, common_token_to_piece(ctx, best_id), max_val};
}

int main(int argc, char **argv) {
    setbuf(stdout, NULL);
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <target_path> <draft_path>\n", argv[0]);
        return 1;
    }

    // 1. 初始化与加载
    common_init();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model* model_tgt = llama_model_load_from_file(argv[1], model_params);
    llama_model* model_dft = llama_model_load_from_file(argv[2], model_params);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 2048;
    ctx_params.n_batch = 512;
    
    // Target 需要 embeddings=true 来提取 hidden states (如果是旧版 llama.cpp 可能不需要显式开启，视版本而定)
    // 新版 llama.cpp 如果要用 extract_hidden_states，通常需要 pool type 或特定设置，但默认 context 应该支持
    ctx_params.embeddings = true; 
    llama_context* ctx_tgt = llama_init_from_model(model_tgt, ctx_params);

    ctx_params.embeddings = false; // Draft 不需要输出 embeddings
    llama_context* ctx_dft = llama_init_from_model(model_dft, ctx_params);

    // 2. 绑定 EAGLE Shared Embedding
    printf("🔗 Binding Embedding Layer...\n");
    llama_context_set_target_embedding_layer(ctx_dft, ctx_tgt);
    const int32_t* d2t_map = llama_model_get_eagle_d2t_map(model_dft);

    // 3. 准备 Prompt
    std::string prompt_str = "Actually, I think";
    std::vector<llama_token> inputs = common_tokenize(ctx_tgt, prompt_str, true);
    
    printf("Prompt: %s\n", prompt_str.c_str());
    printf("--------------------------------------------------------\n");

    // ===================================================================================
    // 阶段 1: Target Model 预热 (Prefill)
    // 目标：处理完 Prompt，拿到最后一个 Token 的 HS，以及预测出的第一个新 Token (T1)
    // ===================================================================================
    
    // Target Decode Prompt
    llama_decode(ctx_tgt, llama_batch_get_one(inputs.data(), inputs.size()));

    // 提取 Prompt 最后一个 Token 的 Hidden States (仿照 simple-logic.cpp)
    std::vector<float> current_hs;
    int last_pos = inputs.size() - 1;
    for (int layer : EAGLE_LAYERS) {
        int32_t dim = 0;
        float* hs = llama_context_extract_hidden_states_from_layer(ctx_tgt, last_pos, layer, &dim);
        if (hs) {
            current_hs.insert(current_hs.end(), hs, hs + dim);
            free(hs);
        } else {
            fprintf(stderr, "❌ Failed extract layer %d\n", layer);
            return 1;
        }
    }

    // 获取 Target 对 T1 的真实预测 (Ground Truth)
    StepResult target_next = sample_greedy(ctx_tgt);
    
    // 当前“输入”给下一轮的 Token 就是 Prompt 的最后一个 Token
    llama_token last_input_token = inputs.back();

    printf("Initialization Complete.\n");
    printf("Prompt Last Token: %d\n", last_input_token);
    printf("Target Predicts T1: '%s' (%d)\n", target_next.token_str.c_str(), target_next.token_id);
    printf("--------------------------------------------------------\n");

    // ===================================================================================
    // 阶段 2: 循环验证 (Step-by-Step Loop)
    // 逻辑：
    //   1. Draft: 输入 (HS_prev, Token_prev) -> 预测 Token_curr
    //   2. Verify: 比较 Draft(Token_curr) vs Target(Token_curr)
    //   3. Target: 输入 (Target_Token_curr) -> 推进一步 -> 产生 HS_curr 和 Token_next
    //   4. 准备下一轮
    // ===================================================================================

    int correct_cnt = 0;
    int top5_hit_cnt = 0;

    // 记录完整句子
    std::vector<llama_token> complete_tokens = inputs; // 初始化为prompt tokens

    for (int step = 0; step < MAX_STEPS; step++) {
        printf("\n[Step %d] \n", step + 1);

        // --- A. Draft 推理 (先预测) ---
        // 设置 Target 上一轮的 HS (即 last_input_token 对应位置的 HS)
        llama_context_set_target_hidden_states(ctx_dft, current_hs.data(), current_hs.size());

        // Draft 输入的是上一步Target预测的 token (不是当前要验证的token)
        printf("   Draft Input: %d ('%s')\n", target_next.token_id, target_next.token_str.c_str());
        llama_batch batch_dft = llama_batch_get_one(&target_next.token_id, 1);
        llama_decode(ctx_dft, batch_dft);

        // 获取 Draft 预测结果
        auto* logits_dft = llama_get_logits(ctx_dft);
        const auto* vocab_dft = llama_model_get_vocab(model_dft);
        int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);

        // 寻找 Draft 的 Top-K 并映射回 Target 空间
        std::vector<std::pair<float, int>> candidates;
        for(int i=0; i<n_vocab_dft; i++) {
            int tgt_id = i + d2t_map[i];
            candidates.push_back({logits_dft[i], tgt_id});
        }
        std::partial_sort(candidates.begin(), candidates.begin() + TOP_K_SHOW, candidates.end(),
                         [](auto a, auto b){ return a.first > b.first; });

        printf("   Draft Top-K Predictions:\n");
        for(int i=0; i<TOP_K_SHOW; i++) {
            std::string s = common_token_to_piece(ctx_tgt, candidates[i].second);
            printf("     [%d] %.4f -> '%s'\n", i, candidates[i].first, s.c_str());
        }

        // --- B. Target 推理 (后验证) ---
        // Target使用上一步的预测结果进行decode
        printf("   Target Decode Input: %d ('%s')\n", target_next.token_id, target_next.token_str.c_str());
        llama_batch batch_tgt = llama_batch_get_one(&target_next.token_id, 1);
        llama_decode(ctx_tgt, batch_tgt);

        // 获取Target的实际输出 (用于与Draft预测比较)
        StepResult actual_target = sample_greedy(ctx_tgt);
        printf("   Target Decode Output: %d ('%s')\n", actual_target.token_id, actual_target.token_str.c_str());

        // 记录到完整句子中
        complete_tokens.push_back(actual_target.token_id);

        // --- C. 验证比较 ---
        bool match = (candidates[0].second == actual_target.token_id);
        if (match) correct_cnt++;

        // 检查Top5命中
        bool top5_hit = false;
        int top5_hit_rank = -1;
        for(int i=0; i<std::min(5, (int)candidates.size()); i++) {
            if (candidates[i].second == actual_target.token_id) {
                top5_hit = true;
                top5_hit_rank = i + 1;
                top5_hit_cnt++;
                break;
            }
        }

        printf("   Verification:\n");
        for(int i=0; i<TOP_K_SHOW; i++) {
            std::string s = common_token_to_piece(ctx_tgt, candidates[i].second);
            printf("     [%d] %.4f -> '%s' %s\n",
                i, candidates[i].first, s.c_str(),
                (candidates[i].second == actual_target.token_id) ? "✅ MATCH" : "");
        }
        printf("   Top-1: %s | Top-5: %s (rank %d)\n",
               match ? "✅ HIT" : "❌ MISS",
               top5_hit ? "✅ HIT" : "❌ MISS",
               top5_hit_rank);

        // --- D. 状态更新 ---
        // 1. 提取刚刚decode位置的 Hidden States (给下一轮Draft用)
        last_pos++; // Context 位置 +1
        current_hs.clear();
        for (int layer : EAGLE_LAYERS) {
            int32_t dim = 0;
            float* hs = llama_context_extract_hidden_states_from_layer(ctx_tgt, 0, layer, &dim);
            if (hs) {
                current_hs.insert(current_hs.end(), hs, hs + dim);
                free(hs);
            } else {
                fprintf(stderr, "❌ Failed extract layer %d at step %d\n", layer, step+1);
                break;
            }
        }

        // 2. 更新下一轮的变量
        last_input_token = target_next.token_id;
        target_next = actual_target; // 下一轮使用Target实际输出的token
    }

    printf("\n========================================\n");
    printf("Result: %d / %d (%.2f%%) Top-1 Accuracy\n", correct_cnt, MAX_STEPS, (float)correct_cnt/MAX_STEPS*100.0);
    printf("Top-5 Hit Rate: %d / %d (%.2f%%)\n", top5_hit_cnt, MAX_STEPS, (float)top5_hit_cnt/MAX_STEPS*100.0);

    // 输出完整句子
    printf("\nComplete Sentence:\n");
    printf("Prompt: %s\n", prompt_str.c_str());
    std::string generated_text = "";
    for (size_t i = inputs.size(); i < complete_tokens.size(); i++) {
        generated_text += common_token_to_piece(ctx_tgt, complete_tokens[i]);
    }
    printf("Generated: %s\n", generated_text.c_str());
    printf("Full Text: %s%s\n", prompt_str.c_str(), generated_text.c_str());

    llama_free(ctx_tgt); llama_free(ctx_dft);
    llama_model_free(model_tgt); llama_model_free(model_dft);
    llama_backend_free();
    return 0;
}
