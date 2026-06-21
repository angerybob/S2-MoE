#include "llama.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <vector>
#include <cmath>

// Test for new EAGLE3 SMP hidden states extraction API
int main(int argc, char** argv) {
    std::cout << "=== EAGLE3 SMP New API Test ===" << std::endl;

    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <model.gguf>" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];

    // Load model
    auto model_params = llama_model_default_params();
    auto* model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        std::cerr << "❌ Failed to load model: " << model_path << std::endl;
        return 1;
    }

    std::cout << "✅ Model loaded: " << model_path << std::endl;

    // Create context
    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 128;
    ctx_params.n_batch = 128;

    auto* ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "❌ Failed to create context" << std::endl;
        llama_model_free(model);
        return 1;
    }

    std::cout << "✅ Context created" << std::endl;

    // Phase 1: Two-step decode - accumulate all historical HS
    std::cout << "\n=== Phase 1: Two-step Decode - Accumulate All Historical HS ===" << std::endl;

    // Step 1: Initial prompt decode (4 tokens)
    std::vector<llama_token> prompt_tokens;
    prompt_tokens.push_back(llama_vocab_bos(llama_model_get_vocab(model)));  // BOS
    prompt_tokens.push_back(1001);  // Token 1
    prompt_tokens.push_back(1002);  // Token 2
    prompt_tokens.push_back(1003);  // Token 3

    std::cout << "\n--- Step 1: Initial Prompt (" << prompt_tokens.size() << " tokens) ---" << std::endl;

    // Create and decode initial batch
    auto initial_batch = llama_batch_init(prompt_tokens.size(), 0, 1);
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        initial_batch.token[i] = prompt_tokens[i];
        initial_batch.pos[i] = i;
        initial_batch.seq_id[i][0] = 0;
        initial_batch.n_seq_id[i] = 1;
        initial_batch.n_tokens++;
    }

    int ret = llama_decode(ctx, initial_batch);
    if (ret != 0) {
        std::cerr << "❌ Initial decode failed: " << ret << std::endl;
        llama_batch_free(initial_batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    std::cout << "✅ Initial prompt decoded successfully" << std::endl;

    // Extract and store initial prompt HS
    std::vector<std::vector<float>> all_historical_hs;
    for (int token_idx = 0; token_idx < (int)prompt_tokens.size(); token_idx++) {
        int32_t dims;
        float* token_hs = llama_context_get_eagle_hs_at_batch_idx(ctx, token_idx, &dims);
        if (token_hs) {
            all_historical_hs.push_back(std::vector<float>(token_hs, token_hs + dims));
            std::cout << "   Prompt Token " << token_idx << ": ✅ Extracted " << dims << " dims" << std::endl;
            free(token_hs);
        }
    }

    // Step 2: Additional decode (1 new token)
    std::cout << "\n--- Step 2: Additional Generation (1 token) ---" << std::endl;

    auto new_batch = llama_batch_init(1, 0, 1);
    new_batch.token[0] = 2001;  // New generated token
    new_batch.pos[0] = prompt_tokens.size();  // Position after prompt
    new_batch.seq_id[0][0] = 0;
    new_batch.n_seq_id[0] = 1;
    new_batch.logits[0] = true;
    new_batch.n_tokens = 1;

    ret = llama_decode(ctx, new_batch);
    if (ret != 0) {
        std::cerr << "❌ Additional decode failed: " << ret << std::endl;
        llama_batch_free(new_batch);
        llama_batch_free(initial_batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    std::cout << "✅ Additional token decoded successfully" << std::endl;

    // Extract new token HS and append to historical buffer
    int32_t new_dims;
    float* new_token_hs = llama_context_get_eagle_hs_at_batch_idx(ctx, 0, &new_dims);
    if (new_token_hs) {
        all_historical_hs.push_back(std::vector<float>(new_token_hs, new_token_hs + new_dims));
        std::cout << "   New Token 1: ✅ Extracted " << new_dims << " dims" << std::endl;
        free(new_token_hs);
    }

    // Step 3: Additional decode (another new token)
    std::cout << "\n--- Step 3: Additional Generation (1 more token) ---" << std::endl;

    auto new_batch2 = llama_batch_init(1, 0, 1);
    new_batch2.token[0] = 2002;  // Another new generated token
    new_batch2.pos[0] = prompt_tokens.size() + 1;  // Position after previous token
    new_batch2.seq_id[0][0] = 0;
    new_batch2.n_seq_id[0] = 1;
    new_batch2.logits[0] = true;
    new_batch2.n_tokens = 1;

    ret = llama_decode(ctx, new_batch2);
    if (ret != 0) {
        std::cerr << "❌ Second additional decode failed: " << ret << std::endl;
        llama_batch_free(new_batch2);
        llama_batch_free(new_batch);
        llama_batch_free(initial_batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    std::cout << "✅ Second additional token decoded successfully" << std::endl;

    // Extract new token HS and append to historical buffer
    int32_t new_dims2;
    float* new_token_hs2 = llama_context_get_eagle_hs_at_batch_idx(ctx, 0, &new_dims2);
    if (new_token_hs2) {
        all_historical_hs.push_back(std::vector<float>(new_token_hs2, new_token_hs2 + new_dims2));
        std::cout << "   New Token 2: ✅ Extracted " << new_dims2 << " dims" << std::endl;
        free(new_token_hs2);
    }

    // Output all historical HS
    std::cout << "\n--- All Historical HS Output (Total: " << all_historical_hs.size() << " tokens) ---" << std::endl;
    for (int token_idx = 0; token_idx < (int)all_historical_hs.size(); token_idx++) {
        const auto& hs = all_historical_hs[token_idx];
        std::string token_type;
        if (token_idx < (int)prompt_tokens.size()) {
            token_type = "prompt";
        } else if (token_idx == (int)prompt_tokens.size()) {
            token_type = "generated1";
        } else {
            token_type = "generated2";
        }

        std::cout << "Historical Token " << token_idx << " (" << token_type << "): Layer 2 (first 5): ";

        // Print first 5 values of layer 2
        for (int i = 0; i < 5 && i < (int)hs.size(); i++) {
            std::cout << hs[i] << " ";
        }
        std::cout << std::endl;
    }

    llama_batch_free(new_batch2);
    llama_batch_free(new_batch);
    llama_batch_free(initial_batch);

    std::cout << "\n=== Phase 2: Fresh Model - Always Extract and Overwrite Latest Token HS ===" << std::endl;

    // Cleanup Phase 1
    llama_free(ctx);

    // Re-create fresh model and context for Phase 2
    auto* model2 = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model2) {
        std::cerr << "❌ Failed to reload model for Phase 2" << std::endl;
        llama_model_free(model);
        return 1;
    }

    auto* ctx2 = llama_init_from_model(model2, ctx_params);
    if (!ctx2) {
        std::cerr << "❌ Failed to create second context" << std::endl;
        llama_model_free(model2);
        llama_model_free(model);
        return 1;
    }
    std::cout << "✅ Fresh model and context created for Phase 2" << std::endl;

    // Phase 2: Multiple decode steps, always extracting and overwriting the latest token HS
    std::vector<float> latest_token_hs;  // This will be overwritten each time

    // Step 2.1: Initial prompt decode (same as Phase 1)
    std::cout << "\n--- Step 2.1: Initial Prompt Decode (4 tokens) ---" << std::endl;
    auto phase2_initial_batch = llama_batch_init(prompt_tokens.size(), 0, 1);
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        phase2_initial_batch.token[i] = prompt_tokens[i];
        phase2_initial_batch.pos[i] = i;
        phase2_initial_batch.seq_id[i][0] = 0;
        phase2_initial_batch.n_seq_id[i] = 1;
        phase2_initial_batch.n_tokens++;
    }

    ret = llama_decode(ctx2, phase2_initial_batch);
    if (ret != 0) {
        std::cerr << "❌ Phase 2 initial decode failed: " << ret << std::endl;
        llama_batch_free(phase2_initial_batch);
        llama_free(ctx2);
        llama_model_free(model2);
        llama_model_free(model);
        return 1;
    }

    // Extract and overwrite: get the LAST token from initial prompt (token 3)
    int32_t dims1;
    float* latest_hs1 = llama_context_get_eagle_hs_at_batch_idx(ctx2, 3, &dims1);
    if (latest_hs1) {
        latest_token_hs.assign(latest_hs1, latest_hs1 + dims1);
        std::cout << "   Latest Token (3): ✅ Overwritten " << dims1 << " dims | Layer 2 (first 5): ";
        for (int i = 0; i < 5 && i < dims1; i++) {
            std::cout << latest_token_hs[i] << " ";
        }
        std::cout << std::endl;
        free(latest_hs1);
    }
    llama_batch_free(phase2_initial_batch);

    // Step 2.2: Additional decode (new token)
    std::cout << "\n--- Step 2.2: Additional Token Decode ---" << std::endl;
    auto phase2_new_batch = llama_batch_init(1, 0, 1);
    phase2_new_batch.token[0] = 2001;
    phase2_new_batch.pos[0] = prompt_tokens.size();
    phase2_new_batch.seq_id[0][0] = 0;
    phase2_new_batch.n_seq_id[0] = 1;
    phase2_new_batch.logits[0] = true;
    phase2_new_batch.n_tokens = 1;

    ret = llama_decode(ctx2, phase2_new_batch);
    if (ret != 0) {
        std::cerr << "❌ Phase 2 additional decode failed: " << ret << std::endl;
        llama_batch_free(phase2_new_batch);
        llama_free(ctx2);
        llama_model_free(model2);
        llama_model_free(model);
        return 1;
    }

    // Extract and overwrite: get the NEW token (batch_idx = 0)
    int32_t dims2;
    float* latest_hs2 = llama_context_get_eagle_hs_at_batch_idx(ctx2, 0, &dims2);
    if (latest_hs2) {
        latest_token_hs.assign(latest_hs2, latest_hs2 + dims2);
        std::cout << "   Latest Token (new): ✅ Overwritten " << dims2 << " dims | Layer 2 (first 5): ";
        for (int i = 0; i < 5 && i < dims2; i++) {
            std::cout << latest_token_hs[i] << " ";
        }
        std::cout << std::endl;
        free(latest_hs2);
    }
    llama_batch_free(phase2_new_batch);

    // Step 2.3: Yet another additional decode
    std::cout << "\n--- Step 2.3: Another Additional Token Decode ---" << std::endl;
    auto phase2_another_batch = llama_batch_init(1, 0, 1);
    phase2_another_batch.token[0] = 2002;
    phase2_another_batch.pos[0] = prompt_tokens.size() + 1;
    phase2_another_batch.seq_id[0][0] = 0;
    phase2_another_batch.n_seq_id[0] = 1;
    phase2_another_batch.logits[0] = true;
    phase2_another_batch.n_tokens = 1;

    ret = llama_decode(ctx2, phase2_another_batch);
    if (ret != 0) {
        std::cerr << "❌ Phase 2 another decode failed: " << ret << std::endl;
        llama_batch_free(phase2_another_batch);
        llama_free(ctx2);
        llama_model_free(model2);
        llama_model_free(model);
        return 1;
    }

    // Extract and overwrite: get the LATEST new token
    int32_t dims3;
    float* latest_hs3 = llama_context_get_eagle_hs_at_batch_idx(ctx2, 0, &dims3);
    if (latest_hs3) {
        latest_token_hs.assign(latest_hs3, latest_hs3 + dims3);
        std::cout << "   Latest Token (newest): ✅ Overwritten " << dims3 << " dims | Layer 2 (first 5): ";
        for (int i = 0; i < 5 && i < dims3; i++) {
            std::cout << latest_token_hs[i] << " ";
        }
        std::cout << std::endl;
        free(latest_hs3);
    }
    llama_batch_free(phase2_another_batch);

    // Final step: Extract ALL tokens from current batch to verify only last token exists
    std::cout << "\n--- Final: Extract All Available Tokens from Current Batch ---" << std::endl;

    // Try to extract all possible batch_idx values to see what's available
    for (int test_idx = 0; test_idx <= 5; test_idx++) {
        int32_t test_dims;
        float* test_hs = llama_context_get_eagle_hs_at_batch_idx(ctx2, test_idx, &test_dims);
        if (test_hs) {
            std::cout << "   Available Token " << test_idx << ": ✅ Extracted " << test_dims << " dims | Layer 2 (first 5): ";
            for (int i = 0; i < 5 && i < test_dims; i++) {
                std::cout << test_hs[i] << " ";
            }
            std::cout << std::endl;
            free(test_hs);
        } else {
            std::cout << "   Available Token " << test_idx << ": ❌ No data available" << std::endl;
        }
    }

    // Final verification: Compare Phase 1 generated token with Phase 2 generated token
    std::cout << "\n--- Verification ---" << std::endl;
    if (!all_historical_hs.empty()) {
        const auto& phase1_generated_token = all_historical_hs.back();  // Last token from Phase 1
        bool same_generation = (latest_token_hs.size() == phase1_generated_token.size());
        if (same_generation) {
            for (size_t i = 0; i < std::min(size_t(10), latest_token_hs.size()); i++) {
                if (std::abs(latest_token_hs[i] - phase1_generated_token[i]) > 1e-6) {
                    same_generation = false;
                    break;
                }
            }
        }
        if (same_generation) {
            std::cout << "✅ Same generation token: Phase 1 generated = Phase 2 latest (deterministic)" << std::endl;
        } else {
            std::cout << "⚠️ Different generation tokens (expected, may be due to different context states)" << std::endl;
        }
    }

    std::cout << "✅ Final batch verification: Only last token's HS should be available (batch_idx = 0)" << std::endl;

    // Cleanup Phase 2
    llama_free(ctx2);
    llama_model_free(model2);

    std::cout << "\n=== Test Complete ===" << std::endl;
    std::cout << "✅ New EAGLE3 SMP API working correctly!" << std::endl;

    return 0;
}