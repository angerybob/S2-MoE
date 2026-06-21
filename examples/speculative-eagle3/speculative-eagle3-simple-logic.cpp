#include "common.h"
#include "llama.h"

#include <vector>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>

// Forward declarations
void print_token_info(llama_context* ctx, llama_token token, const char* prefix);
void print_top_preds(llama_context* ctx, const float* logits, int vocab_size, const int32_t* d2t_map, const char* prefix);

// 辅助函数：打印Token信息
void print_token_info(llama_context* ctx, llama_token token, const char* prefix) {
    std::string piece = common_token_to_piece(ctx, token);
    printf("%sToken %d = '%s'\n", prefix, token, piece.c_str());
}

void print_top_preds(llama_context* ctx, const float* logits, int vocab_size, const int32_t* d2t_map, const char* prefix) {
    std::vector<std::pair<float, int>> preds;
    for(int i=0; i<vocab_size; i++) preds.push_back({logits[i], i + d2t_map[i]});
    std::partial_sort(preds.begin(), preds.begin()+5, preds.end(), [](auto a, auto b){return a.first > b.first;});

    for(int i=0; i<5; i++) {
        std::string s = common_token_to_piece(ctx, preds[i].second);
        printf("%s [%d] '%s' (%.4f)\n", prefix, preds[i].second, s.c_str(), preds[i].first);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <target_model> <draft_model>\n", argv[0]);
        return 1;
    }

    const char* target_model_path = argv[1];
    const char* draft_model_path = argv[2];

    printf("=== EAGLE3 Shared Embedding Logic Verification ===\n");
    printf("Flow: \n");
    printf("1. Target: Decode Prompt -> Get HiddenStates(t4) & Predict t5\n");
    printf("2. Target: Decode t5 -> Get Truth t6 (Reference)\n");
    printf("3. Draft:  Input [HS(t4) + TokenID(t5)] -> Uses Shared Emb Layer -> Predict t6\n");
    printf("4. Verify: Compare Draft t6 with Target Truth t6\n\n");

    // 1. 初始化
    common_init();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model* model_tgt = llama_model_load_from_file(target_model_path, model_params);
    llama_model* model_draft = llama_model_load_from_file(draft_model_path, model_params);

    if (!model_tgt || !model_draft) {
        fprintf(stderr, "❌ Failed to load models\n");
        return 1;
    }

    // 2. 创建Context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.n_batch = 1024;
    
    // Target Model: 正常运行
    ctx_params.embeddings = false; // 此时Target只需生成logits，不需要显式提取embedding向量给Draft了
    llama_context* ctx_tgt = llama_init_from_model(model_tgt, ctx_params);

    // Draft Model: 依靠共享层
    ctx_params.embeddings = false; 
    llama_context* ctx_draft = llama_init_from_model(model_draft, ctx_params);

    // 3. 关键绑定：设置Draft共享Target的Embedding Layer
    printf("🔗 Binding Draft Model to Target Embedding Layer...\n");
    llama_context_set_target_embedding_layer(ctx_draft, ctx_tgt);
    if (!llama_context_get_shared_token_embd(ctx_draft)) {
        fprintf(stderr, "❌ Failed to bind shared embedding layer\n");
        return 1;
    }
    printf("✅ Bind successful.\n");

    // 4. 获取必要参数
    const int32_t* d2t_map = llama_model_get_eagle_d2t_map(model_draft);
    int tgt_vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
    int draft_vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model_draft));

    if (!d2t_map) { fprintf(stderr, "❌ Missing d2t_map\n"); return 1; }

    // 5. 准备Prompt
    std::string prompt_str = "I can help you with your";
    std::vector<llama_token> prompt_tokens = common_tokenize(ctx_tgt, prompt_str, true);
    
    printf("\nPrompt: \"%s\" (%zu tokens)\n", prompt_str.c_str(), prompt_tokens.size());

    // =================================================================
    // PHASE 1: Target Model Decode Prompt
    // 目标：
    // 1. 生成 t4 的 Hidden States (给 Draft 用)
    // 2. 生成 t5 (作为 Draft 的输入 Token)
    // =================================================================
    printf("\n=== PHASE 1: Target Decode Prompt ===\n");
    
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx_tgt, batch) != 0) return 1;

    // 1.1 提取 Hidden States (t4, 最后一个 prompt token)
    std::vector<float> hidden_states_t4;
    std::vector<int> eagle_layers = {2, 18, 33}; // 请确保层数与 config.json 一致
    int last_token_idx = prompt_tokens.size() - 1; 

    for (int layer : eagle_layers) {
        int32_t dim = 0;
        float* hs = llama_context_extract_hidden_states_from_layer(ctx_tgt, last_token_idx, layer, &dim);
        if (hs) {
            hidden_states_t4.insert(hidden_states_t4.end(), hs, hs + dim);
            free(hs);
        } else {
            fprintf(stderr, "❌ Failed extract layer %d\n", layer); return 1;
        }
    }
    printf("✅ Extracted Hidden States at t4 (Size: %zu)\n", hidden_states_t4.size());

    printf("Hidden States Stats:\n");
    float hs_min = 1e9, hs_max = -1e9, hs_sum = 0;
    for(float v : hidden_states_t4) {
        if(v < hs_min) hs_min = v;
        if(v > hs_max) hs_max = v;
        hs_sum += fabs(v);
    }
    printf("  Min: %.4f, Max: %.4f, Avg Abs: %.4f\n", 
        hs_min, hs_max, hs_sum / hidden_states_t4.size());

    if (hs_sum < 1.0) {
        printf("⚠️ WARNING: Hidden states seem empty or near zero!\n");
    }

    // 1.2 获取 Target 预测的 t5 (Token ID)
    auto* logits_prompt = llama_get_logits(ctx_tgt);
    int token_t5 = 0;
    float max_val = -1e9;
    for(int i=0; i<tgt_vocab_size; i++) {
        if(logits_prompt[i] > max_val) { max_val = logits_prompt[i]; token_t5 = i; }
    }
    print_token_info(ctx_tgt, token_t5, "Target predicts t5: ");


    // =================================================================
    // PHASE 2: Target Model Decode t5 (生成 Ground Truth)
    // 目标：
    // 1. 让 Target 往前走一步
    // 2. 获取 t6 的真实预测结果，用于验证 Draft 是否正确
    // =================================================================
    printf("\n=== PHASE 2: Target Decode t5 (Get Ground Truth) ===\n");
    
    llama_batch batch_t5 = llama_batch_get_one(&token_t5, 1);
    if (llama_decode(ctx_tgt, batch_t5) != 0) return 1;

    // 获取 Target 预测的 t6 (Ground Truth)
    auto* logits_t5 = llama_get_logits(ctx_tgt);
    int token_t6_true = 0;
    max_val = -1e9;
    for(int i=0; i<tgt_vocab_size; i++) {
        if(logits_t5[i] > max_val) { max_val = logits_t5[i]; token_t6_true = i; }
    }
    print_token_info(ctx_tgt, token_t6_true, "Target Reference t6: ");


    // =================================================================
    // PHASE 3: Draft Model Prediction (EAGLE3 Logic)
    // 输入: 
    //   1. HS(t4) -> Set via API
    //   2. Token(t5) -> Pass via Batch (Draft uses shared weights to lookup)
    // 输出: t6 prediction
    // =================================================================
    printf("\n=== PHASE 3: Draft Model Prediction (Shared Embedding) ===\n");

    // 3.1 设置 Hidden States
    llama_context_set_target_hidden_states(ctx_draft, hidden_states_t4.data(), hidden_states_t4.size());

    // 3.2 构造 Batch：直接传入 token_t5
    // 关键点：这里不再传 dummy token，而是传真实的 t5
    // 因为 Draft Model 现在有能力处理 Token ID 了（通过共享权重）
    llama_batch batch_draft = llama_batch_get_one(&token_t5, 1);
    
    // 3.3 Decode
    if (llama_decode(ctx_draft, batch_draft) != 0) {
        fprintf(stderr, "❌ Draft decode failed\n");
        return 1;
    }

    // 3.4 获取 Draft 预测并映射回 Target 空间
    auto* logits_draft = llama_get_logits(ctx_draft);
    
    // 获取 Top-K 候选
    int K = 10;
    std::vector<std::pair<float, int>> draft_cands;
    draft_cands.reserve(draft_vocab_size);
    
    for(int i=0; i<draft_vocab_size; i++) {
        // Eagle 映射逻辑: target_id = draft_id + map[draft_id]
        int tgt_id = i + d2t_map[i];
        draft_cands.push_back({logits_draft[i], tgt_id});
    }
    
    // 排序
    std::partial_sort(draft_cands.begin(), draft_cands.begin() + K, draft_cands.end(),
                      [](const auto& a, const auto& b){ return a.first > b.first; });

    printf("Draft Model Top-%d predictions (t6 candidates):\n", K);
    bool found_correct = false;
    int found_rank = -1;

    for(int i=0; i<K; i++) {
        int id = draft_cands[i].second;
        std::string s = common_token_to_piece(ctx_tgt, id);
        bool match = (id == token_t6_true);
        
        if(match) {
            found_correct = true;
            found_rank = i + 1;
        }
        
        printf("  [%d] Token %d = '%s' (logit: %.4f) %s\n", 
            i+1, id, s.c_str(), draft_cands[i].first, match ? "✅ MATCH" : "");
    }


    // =================================================================
    // PHASE 4: Final Verdict
    // =================================================================
    printf("\n=== PHASE 4: Verification Result ===\n");
    if (found_correct) {
        printf("🎉 SUCCESS: EAGLE3 Logic Verified!\n");
        printf("Details:\n");
        printf("  - Input HS: From '...with'\n");
        printf("  - Input Token: '%s' (ID %d)\n", common_token_to_piece(ctx_tgt, token_t5).c_str(), token_t5);
        printf("  - Shared Embedding Layer: Used to lookup ID %d\n", token_t5);
        printf("  - Target Expects: '%s' (ID %d)\n", common_token_to_piece(ctx_tgt, token_t6_true).c_str(), token_t6_true);
        printf("  - Draft Predicted: Found at Rank %d\n", found_rank);
    } else {
        printf("❌ FAILURE: Draft did not predict the correct token in Top-%d.\n", K);
        printf("  - Check layers configuration (currently: 2, 18, 33)\n");
        printf("  - Check d2t_map validity\n");
    }

    // Cleanup
    llama_free(ctx_tgt); llama_free(ctx_draft);
    llama_model_free(model_tgt); llama_model_free(model_draft);
    llama_backend_free();
    return 0;
}
