#include "common.h"
#include "llama.h"
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <iomanip>

// Forward declarations
int get_argmax(const float* logits, int n_vocab);
int get_rank(const float* logits, int n_vocab, int token_id);
void clean_string(std::string& s);
std::vector<float> extract_hs_helper(llama_context* ctx, int batch_idx, const std::vector<int>& layers);

// 测试配置
const std::vector<std::string> TEST_PROMPTS = {
    // 1. Common Conversational
    "I can help you with your",
    "The weather today is very",
    "Hello, how can I help",
    "Once upon a time, there was a",
    "Please explain the theory of",
    // 2. Coding & Technical
    "def fibonacci(n):",
    "#include <iostream>",
    "SELECT * FROM users WHERE",
    "import numpy as",
    "const express = require(",
    // 3. Q&A / Instruction
    "What is the capital of France?",
    "How to bake a chocolate",
    "Translate the following sentence to",
    "The quick brown fox jumps over",
    "To be or not to be, that is",
    // 4. Reasoning / Math
    "The square root of 144 is",
    "If x = 5 and y = 10, then x + y =",
    "Step 1: Open the box. Step 2:",
    "A triangle has three",
    "Newton's first law of motion states",
    // 5. Completions
    "Thank you for your",
    "Best regards,",
    "In conclusion, we can say",
    "However, on the other hand,",
    "It is important to note that",
    // 4. Everyday Conversations
    "Good morning, the weather today",
    "What should we have for",
    "Any plans for this",
    "I think this idea is quite",
    "How have you been",
    "Let's go take a",
    "Drive safely on the",
    "Remember to drink more",
    "Wish you good luck with your",
    "Have a wonderful",
    // 5. Half-sentence Patterns
    "If that's the case, then",
    "Taking this factor into",
    "Generally speaking, I",
    "By the way, I",
    "Actually, I think",
    "In other words, we",
    "More importantly, the",
    "Obviously, this is",
    "Relatively speaking, it",
    // 6. Daily Life Scenarios
    "What should we eat for",
    "I'm feeling pretty good",
    "Do you need any help with",
    "I'll be there in just a",
    "Please wait for me a",
    "Everyone has already",
    "Time really flies when",
    "Are you free tonight to",
    "See you tomorrow morning",
    // 7. Common Expressions
    "I think this is pretty good",
    "This problem is quite",
    "Let me think of a way to",
    "In principle, we should",
    "From a practical point of view",
    "Based on my personal",
    "Under normal circumstances, it",
    "Generally speaking, this should"
    // (你可以自行扩展到 100 个)
};

struct TestResult {
    std::string prompt;
    std::string t5_str;
    std::string t6_truth_str;
    std::string t6_draft_str;
    int draft_rank;
};

// 辅助函数
int get_argmax(const float* logits, int n_vocab) {
    int max_i = 0;
    float max_v = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > max_v) {
            max_v = logits[i];
            max_i = i;
        }
    }
    return max_i;
}

// 计算某个 token 在 logits 分布中的排名（0为最高）
int get_rank(const float* logits, int n_vocab, int token_id) {
    float target_val = logits[token_id];
    int rank = 0;
    for (int i = 0; i < n_vocab; i++) {
        if (logits[i] > target_val) {
            rank++;
        }
    }
    return rank;
}

void clean_string(std::string& s) {
    std::replace(s.begin(), s.end(), '\n', ' ');
    std::replace(s.begin(), s.end(), '\r', ' ');
}

// 封装提取 HS 的逻辑，与 Code 1 保持一致
std::vector<float> extract_hs_helper(llama_context* ctx, int batch_idx, const std::vector<int>& layers) {
    std::vector<float> res;
    for (int layer : layers) {
        int32_t dim = 0;
        float* ptr = llama_context_extract_hidden_states_from_layer(ctx, batch_idx, layer, &dim);
        if (ptr) {
            res.insert(res.end(), ptr, ptr + dim);
            free(ptr);
        }
    }
    return res;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <target_model> <draft_model>\n", argv[0]);
        return 1;
    }

    common_init();
    llama_backend_init();
    
    // Load Models
    llama_model_params mp = llama_model_default_params();
    llama_model* m_tgt = llama_model_load_from_file(argv[1], mp);
    llama_model* m_dft = llama_model_load_from_file(argv[2], mp);

    if (!m_tgt || !m_dft) return 1;

    // Context Setup
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048; 
    cp.n_batch = 2048;

    cp.embeddings = true; // Target need HS
    llama_context* ctx_tgt = llama_init_from_model(m_tgt, cp);
    cp.embeddings = false;
    llama_context* ctx_dft = llama_init_from_model(m_dft, cp);

    // EAGLE Setup
    llama_context_set_draft_context(ctx_dft, true);
    llama_context_set_target_embedding_layer(ctx_dft, ctx_tgt);
    const int32_t* d2t_map = llama_model_get_eagle_d2t_map(m_dft);
    
    int n_vocab_tgt = llama_vocab_n_tokens(llama_model_get_vocab(m_tgt));
    int n_vocab_dft = llama_vocab_n_tokens(llama_model_get_vocab(m_dft));
    const std::vector<int> eagle_layers = {2, 18, 33}; // 请确保与你的模型匹配

    // Stats
    int top1_hits = 0;
    int top5_hits = 0;
    int total_tests = TEST_PROMPTS.size();

    printf("\n🚀 Starting EAGLE3 Logic Verification...\n");
    printf("%-30s | %-10s | %-10s | %-6s | %-15s\n", "Prompt", "Input(T5)", "Truth(T6)", "Rank", "Draft Pred");
    printf(std::string(85, '-').c_str());
    printf("\n");

    for (int i = 0; i < total_tests; i++) {
        std::string prompt = TEST_PROMPTS[i];

        // 【修正 1】每次循环必须清除 KV Cache，否则状态会累积导致 Context Full
        llama_memory_clear(llama_get_memory(ctx_tgt), true);
        llama_memory_clear(llama_get_memory(ctx_dft), true);

        // ==========================================
        // PHASE 1: Target Prefill (Prompt)
        // ==========================================
        std::vector<llama_token> tokens = common_tokenize(ctx_tgt, prompt, true); // true = add special token
        
        llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
        if (llama_decode(ctx_tgt, batch) != 0) {
            fprintf(stderr, "Decode failed\n");
            continue;
        }

        // 1. 获取 HS (来自 Prompt 最后一个 token)
        std::vector<float> hs = extract_hs_helper(ctx_tgt, tokens.size() - 1, eagle_layers);
        if (hs.empty()) {
            fprintf(stderr, "HS extraction failed. Check layer indices.\n");
            continue;
        }

        // 2. 获取 T5 (Input Token for Draft) - 来自 Target 对 Prompt 的预测
        // 注意：get_logits 返回的是当前 batch 所有位置的 logits
        // tokens.size()-1 是 batch 中的最后一个位置
        float* logits_ptr = llama_get_logits(ctx_tgt);
        float* last_token_logits = logits_ptr + (tokens.size() - 1) * n_vocab_tgt;
        int t5 = get_argmax(last_token_logits, n_vocab_tgt);

        // ==========================================
        // PHASE 2: Draft Prediction (One Step)
        // ==========================================
        // 设置 Target HS 给 Draft 模型
        llama_context_set_target_hidden_states(ctx_dft, hs.data(), hs.size());
        
        // 运行 Draft 模型 (Input: t5)
        llama_batch dft_batch = llama_batch_get_one(&t5, 1);
        llama_decode(ctx_dft, dft_batch);

        // 获取 Draft 预测 (t6_draft)
        int t6_draft_idx = get_argmax(llama_get_logits(ctx_dft), n_vocab_dft);
        int t6_draft_mapped = t6_draft_idx + d2t_map[t6_draft_idx]; // Map back to Target ID

        // ==========================================
        // PHASE 3: Verification (Target runs Forward)
        // ==========================================
        // 【修正 2】不要单独跑 Truth 生成，而是构建 Verify Batch
        // 构造 batch: [t5, t6_draft]
        // t5 的 logits 将告诉我们 t6 到底应该是什么 (Truth)
        // 同时也用于计算 t6_draft 的 rank
        std::vector<llama_token> verify_tokens = {t5, t6_draft_mapped};
        
        llama_batch verify_batch = llama_batch_init(verify_tokens.size(), 0, 1);
        verify_batch.n_tokens = verify_tokens.size();

        for(int k = 0; k < (int)verify_tokens.size(); k++) {
            verify_batch.token[k] = verify_tokens[k];
            // 【修正 3】位置必须紧接 Prompt 之后
            verify_batch.pos[k] = tokens.size() + k; 
            verify_batch.n_seq_id[k] = 1;
            verify_batch.seq_id[k][0] = 0;
            verify_batch.logits[k] = true;
        }

        // 运行 Target 进行验证
        llama_decode(ctx_tgt, verify_batch);

        // ==========================================
        // PHASE 4: Compare & Check
        // ==========================================
        // 我们只关心 verify_batch[0] (即 t5) 产生的 logits
        // 因为 t5 -> 预测 -> t6
        float* verify_logits = llama_get_logits(ctx_tgt); // 指向 batch 的开始
        // index 0 对应的 logits
        float* t5_logits = verify_logits + (0 * n_vocab_tgt);

        // 真正的 Truth (Target 在看到 t5 后想输出什么)
        int t6_truth = get_argmax(t5_logits, n_vocab_tgt);

        // 计算 Draft 预测的 Rank
        int rank = get_rank(t5_logits, n_vocab_tgt, t6_draft_mapped);

        // 释放 batch
        llama_batch_free(verify_batch);

        // ==========================================
        // Stats & Print
        // ==========================================
        if (rank == 0) top1_hits++;
        if (rank < 5) top5_hits++;

        std::string prompt_short = prompt.length() > 28 ? prompt.substr(0, 28) + ".." : prompt;
        std::string t5_s = common_token_to_piece(ctx_tgt, t5);
        std::string t6_truth_s = common_token_to_piece(ctx_tgt, t6_truth);
        std::string t6_draft_s = common_token_to_piece(ctx_tgt, t6_draft_mapped);
        
        clean_string(t5_s); clean_string(t6_truth_s); clean_string(t6_draft_s);

        printf("%-30s | %-10s | %-10s | %-6d | %-15s\n",
               prompt_short.c_str(), t5_s.c_str(), t6_truth_s.c_str(),
               rank + 1, t6_draft_s.c_str());
        fflush(stdout);
    }

    printf(std::string(85, '-').c_str());
    printf("\n📊 SUMMARY\n");
    printf("Top-1 Acc: %.2f%%\n", (float)top1_hits/total_tests*100.0);
    printf("Top-5 Acc: %.2f%%\n", (float)top5_hits/total_tests*100.0);

    llama_free(ctx_tgt); llama_free(ctx_dft);
    llama_model_free(m_tgt); llama_model_free(m_dft);
    llama_backend_free();
    return 0;
}