#include "common.h"
#include "llama.h"
#include <vector>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cmath>

// Forward declaration
void print_top(llama_context* ctx, const float* logits, const int32_t* map, int n_vocab, const char* tag);

void print_top(llama_context* ctx, const float* logits, const int32_t* map, int n_vocab, const char* tag) {
    std::vector<std::pair<float, int>> preds;
    for(int i=0; i<n_vocab; i++) preds.push_back({logits[i], i + map[i]});
    std::partial_sort(preds.begin(), preds.begin()+5, preds.end(), [](auto a, auto b){return a.first > b.first;});
    
    printf("\n--- %s ---\n", tag);
    for(int i=0; i<5; i++) {
        printf("  [%d] '%s' (%.4f)\n", preds[i].second, common_token_to_piece(ctx, preds[i].second).c_str(), preds[i].first);
    }
}

int main(int argc, char **argv) {
    if (argc < 3) return 1;
    common_init();
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    llama_model* m_tgt = llama_model_load_from_file(argv[1], mp);
    llama_model* m_dft = llama_model_load_from_file(argv[2], mp);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 1024;
    
    cp.embeddings = true;
    llama_context* ctx_tgt = llama_init_from_model(m_tgt, cp);
    cp.embeddings = false;
    llama_context* ctx_dft = llama_init_from_model(m_dft, cp);

    // Bind Shared Embedding
    llama_context_set_target_embedding_layer(ctx_dft, ctx_tgt);
    
    const int32_t* d2t = llama_model_get_eagle_d2t_map(m_dft);
    int d_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m_dft));

    // 1. Target Run
    std::string prompt = "I can help you with";
    std::vector<llama_token> tokens = common_tokenize(ctx_tgt, prompt, true);
    llama_decode(ctx_tgt, llama_batch_get_one(tokens.data(), tokens.size()));
    
    // Extract HS
    std::vector<float> hs_real;
    std::vector<int> layers = {2, 18, 33}; 
    for(int l : layers) {
        int dim=0;
        float* ptr = llama_context_extract_hidden_states_from_layer(ctx_tgt, tokens.size()-1, l, &dim);
        hs_real.insert(hs_real.end(), ptr, ptr+dim);
        free(ptr);
    }
    std::vector<float> hs_zero(hs_real.size(), 0.0f);

    // Get t5
    auto* l_tgt = llama_get_logits(ctx_tgt);
    int t5 = 0; 
    float max_v = -1e9;
    int tgt_vocab = llama_vocab_n_tokens(llama_model_get_vocab(m_tgt));
for(int i=0; i<tgt_vocab; i++) if(l_tgt[i]>max_v) {max_v=l_tgt[i]; t5=i;}
    printf("Target predicts t5: '%s'\n", common_token_to_piece(ctx_tgt, t5).c_str());

    // 2. Draft Tests

    // TEST A: Only HS (Use Dummy Token 0 as Embedding Input)
    // 这里的逻辑是：如果 Embedding 是无关紧要的 dummy，Draft 单靠 HS 会预测什么？
    // 注意：Token 0 通常是 <|endoftext|> 或类似的特殊符，embedding 往往非零但无具体语义
    llama_context_set_target_hidden_states(ctx_dft, hs_real.data(), hs_real.size());
    int token_dummy = 0; 
    llama_decode(ctx_dft, llama_batch_get_one(&token_dummy, 1));
    print_top(ctx_tgt, llama_get_logits(ctx_dft), d2t, d_vocab, "TEST A: Only HS (Embedding=Dummy)");

    // TEST B: Only Embedding (Zero HS)
    // 这里的逻辑是：如果没有上下文，Draft 看到 ' that' 会预测什么？
    llama_context_set_target_hidden_states(ctx_dft, hs_zero.data(), hs_zero.size());
    llama_decode(ctx_dft, llama_batch_get_one(&t5, 1));
    print_top(ctx_tgt, llama_get_logits(ctx_dft), d2t, d_vocab, "TEST B: Only Embedding (HS=Zero)");

    // TEST C: Combined (Real HS + Real Embedding)
    // 这是标准流程
    llama_context_set_target_hidden_states(ctx_dft, hs_real.data(), hs_real.size());
    llama_decode(ctx_dft, llama_batch_get_one(&t5, 1));
    print_top(ctx_tgt, llama_get_logits(ctx_dft), d2t, d_vocab, "TEST C: Combined (Standard)");

    // TEST D: Verify HS Integrity
    // 检查提取出的 HS 数值是否正常
    float hs_mean = 0, hs_std = 0;
    for(float v : hs_real) hs_mean += v;
    hs_mean /= hs_real.size();
    for(float v : hs_real) hs_std += (v-hs_mean)*(v-hs_mean);
    hs_std = sqrt(hs_std / hs_real.size());
    printf("\nHS Stats: Mean=%.4f, Std=%.4f\n", hs_mean, hs_std);

    return 0;
}
