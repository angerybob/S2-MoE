#include "common.h"
#include "llama.h"
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <algorithm>

// Forward declaration
void print_token(llama_context* ctx, llama_token token, const char* prefix);

// 打印 Token
void print_token(llama_context* ctx, llama_token token, const char* prefix) {
    printf("%sToken %d = '%s'\n", prefix, token, common_token_to_piece(ctx, token).c_str());
}

int main(int argc, char **argv) {
    if (argc < 3) return 1;

    common_init();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    llama_model* model_tgt = llama_model_load_from_file(argv[1], model_params);
    llama_model* model_draft = llama_model_load_from_file(argv[2], model_params);

    // 1. 创建 Context (Target 必须开启 embeddings 以便我们进行对比验证)
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 1024;
    ctx_params.embeddings = true; // 开启用于诊断
    llama_context* ctx_tgt = llama_init_from_model(model_tgt, ctx_params);

    ctx_params.embeddings = false;
    llama_context* ctx_draft = llama_init_from_model(model_draft, ctx_params);

    // 绑定共享层
    llama_context_set_target_embedding_layer(ctx_draft, ctx_tgt);
    
    // 准备数据
    const int32_t* d2t_map = llama_model_get_eagle_d2t_map(model_draft);
    int n_embd = llama_model_n_embd(model_tgt);
    
    // Prompt: "I can help you with"
    std::vector<llama_token> tokens = common_tokenize(ctx_tgt, "I can help you with", true);
    
    // --- Phase 1: Target Run ---
    llama_decode(ctx_tgt, llama_batch_get_one(tokens.data(), tokens.size()));

    // 提取 Hidden States (Layer 2, 18, 33)
    std::vector<float> hs_t4;
    std::vector<int> layers = {2, 18, 33};
    for(int l : layers) {
        int dim=0;
        float* ptr = llama_context_extract_hidden_states_from_layer(ctx_tgt, tokens.size()-1, l, &dim);
        hs_t4.insert(hs_t4.end(), ptr, ptr+dim);
        free(ptr);
    }
    
    // 获取 t5 预测
    auto* logits = llama_get_logits(ctx_tgt);
    int tgt_vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
    int t5 = 0;
    for(int i=0; i<tgt_vocab_size; i++) if(logits[i] > logits[t5]) t5 = i;
    print_token(ctx_tgt, t5, "Target predicts t5: ");

    // --- Phase 2: Diagnosis (The Smoking Gun) ---
    printf("\n=== DIAGNOSIS: Embedding Consistency Check ===\n");
    
    // 1. 让 Target 跑 t5，提取“计算图输出的 Embedding”
    llama_decode(ctx_tgt, llama_batch_get_one(&t5, 1));
    float* emb_calculated = llama_get_embeddings_ith(ctx_tgt, 0);

    // 2. 从共享权重中直接由 CPU 读取“原始 Embedding”
    // 注意：这里需要访问底层 tensor，我们用一个近似方法：
    // 让 Draft Model 跑一次 t5，如果机制正常，它内部就在用原始权重
    // 但为了对比数值，我们需要手动模拟查表（假设权重在 CPU）
    // 由于 llama.cpp API 限制，我们通过观察 Draft 行为来判断，或者直接看数值差异原因。
    
    // 这里我们做一个假设验证：
    // 如果 Qwen2 架构使用了 scaling (sqrt(d_model))，那么 calculated 会比 raw 大很多。
    
    double sum_sq = 0.0;
    for(int i=0; i<n_embd; i++) sum_sq += emb_calculated[i]*emb_calculated[i];
    double norm_calc = sqrt(sum_sq);
    printf("Calculated Embedding Norm: %.4f\n", norm_calc);
    
    // 典型 Raw Embedding 的 Norm 通常在 1.0 附近或者 sqrt(d) 附近
    // 如果这个值非常大（比如 50.0）或者非常小，能说明问题。
    
    // --- Phase 3: Draft Prediction ---
    printf("\n=== Draft Prediction ===\n");
    llama_context_set_target_hidden_states(ctx_draft, hs_t4.data(), hs_t4.size());
    
    // 关键：Draft 跑 t5
    llama_decode(ctx_draft, llama_batch_get_one(&t5, 1));
    
    auto* d_logits = llama_get_logits(ctx_draft);
    int draft_vocab_size = llama_vocab_n_tokens(llama_model_get_vocab(model_draft));
    std::vector<std::pair<float, int>> preds;
    for(int i=0; i<draft_vocab_size; i++) preds.push_back({d_logits[i], i + d2t_map[i]});
    std::partial_sort(preds.begin(), preds.begin()+5, preds.end(), [](auto a, auto b){return a.first > b.first;});

    for(int i=0; i<5; i++) {
        print_token(ctx_tgt, preds[i].second, "Draft: ");
    }
    
    // --- Phase 4: Verification ---
    // 获取 Target t6 真值 (已经在 Phase 2 decode 过了)
    auto* t_logits = llama_get_logits(ctx_tgt);
    int t6 = 0;
    for(int i=0; i<tgt_vocab_size; i++) if(t_logits[i] > t_logits[t6]) t6 = i;
    print_token(ctx_tgt, t6, "Truth t6: ");
    
    return 0;
}