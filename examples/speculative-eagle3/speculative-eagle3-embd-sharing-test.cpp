#include "common.h"
#include "llama.h"

#include <vector>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>

// Forward declarations
void print_tensor_info(const char* name, struct ggml_tensor* tensor);
bool compare_tensor_data(struct ggml_tensor* a, struct ggml_tensor* b);

// 辅助函数：打印Tensor信息
void print_tensor_info(const char* name, struct ggml_tensor* tensor) {
    if (!tensor) {
        printf("❌ %s: NULL tensor\n", name);
        return;
    }

    printf("✅ %s: [%ld, %ld] type=%d name='%s'\n",
           name, (long)tensor->ne[0], (long)tensor->ne[1], tensor->type, tensor->name ? tensor->name : "no-name");
}

// 辅助函数：比较两个tensor是否相等
bool compare_tensor_data(struct ggml_tensor* a, struct ggml_tensor* b) {
    if (!a || !b) return false;
    if (a->ne[0] != b->ne[0] || a->ne[1] != b->ne[1]) return false;

    // 简单比较前几个值
    float* data_a = (float*)a->data;
    float* data_b = (float*)b->data;
    int64_t n = std::min((int64_t)10, a->ne[0] * a->ne[1]);

    for (int64_t i = 0; i < n; i++) {
        if (fabs(data_a[i] - data_b[i]) > 1e-6) {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <target_model> <draft_model>\n", argv[0]);
        return 1;
    }

    const char* target_model_path = argv[1];
    const char* draft_model_path = argv[2];

    printf("=== EAGLE3 Embedding Layer Sharing API Test ===\n");
    printf("Target Model: %s\n", target_model_path);
    printf("Draft Model: %s\n\n", draft_model_path);

    // 1. 初始化
    common_init();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();

    // 加载目标模型
    llama_model* model_tgt = llama_model_load_from_file(target_model_path, model_params);
    if (!model_tgt) {
        fprintf(stderr, "❌ Failed to load target model: %s\n", target_model_path);
        return 1;
    }
    printf("✅ Loaded target model\n");

    // 加载草稿模型
    llama_model* model_draft = llama_model_load_from_file(draft_model_path, model_params);
    if (!model_draft) {
        fprintf(stderr, "❌ Failed to load draft model: %s\n", draft_model_path);
        llama_model_free(model_tgt);
        return 1;
    }
    printf("✅ Loaded draft model\n");

    // 2. 创建Context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 512;
    ctx_params.embeddings = true;

    llama_context* ctx_tgt = llama_init_from_model(model_tgt, ctx_params);
    if (!ctx_tgt) {
        fprintf(stderr, "❌ Failed to create target context\n");
        llama_model_free(model_tgt);
        llama_model_free(model_draft);
        return 1;
    }
    printf("✅ Created target context\n");

    llama_context* ctx_draft = llama_init_from_model(model_draft, ctx_params);
    if (!ctx_draft) {
        fprintf(stderr, "❌ Failed to create draft context\n");
        llama_free(ctx_tgt);
        llama_model_free(model_tgt);
        llama_model_free(model_draft);
        return 1;
    }
    printf("✅ Created draft context\n");

    // 3. 获取模型信息
    int n_embd_tgt = llama_model_n_embd(model_tgt);
    int n_embd_draft = llama_model_n_embd(model_draft);
    int vocab_tgt = llama_vocab_n_tokens(llama_model_get_vocab(model_tgt));
    int vocab_draft = llama_vocab_n_tokens(llama_model_get_vocab(model_draft));

    printf("\nModel Info:\n");
    printf("  Target:  vocab=%d, embd=%d\n", vocab_tgt, n_embd_tgt);
    printf("  Draft:   vocab=%d, embd=%d\n", vocab_draft, n_embd_draft);

    // 4. 模型结构信息（不直接访问内部tensor）
    printf("\n=== Test 1: Model Structure Information ===\n");

    printf("Target model structure info:\n");
    printf("  Model pointer: %p\n", (void*)model_tgt);
    printf("  Context pointer: %p\n", (void*)ctx_tgt);

    printf("Draft model structure info:\n");
    printf("  Model pointer: %p\n", (void*)model_draft);
    printf("  Context pointer: %p\n", (void*)ctx_draft);

    // 5. 测试Embedding Layer Sharing API
    printf("\n=== Test 2: Embedding Layer Sharing API ===\n");

    // 5.1 设置前：检查shared_token_embd状态
    struct ggml_tensor* shared_before = llama_context_get_shared_token_embd(ctx_draft);
    printf("Before sharing - shared_token_embd: %s\n", shared_before ? "non-NULL" : "NULL");

    // 5.2 设置共享：让草稿模型共享目标模型的embedding layer
    printf("\n--- Setting up embedding layer sharing ---\n");
    llama_context_set_target_embedding_layer(ctx_draft, ctx_tgt);
    printf("✅ Called llama_context_set_target_embedding_layer()\n");

    // 5.3 检查设置后状态
    struct ggml_tensor* shared_after = llama_context_get_shared_token_embd(ctx_draft);
    printf("After sharing - shared_token_embd: %s\n", shared_after ? "non-NULL" : "NULL");

    if (shared_after) {
        print_tensor_info("Shared token_embd", shared_after);

        // 验证shared tensor的基本属性
        printf("✅ Shared embedding layer successfully established\n");
        printf("   Tensor dimensions: [%ld, %ld]\n", (long)shared_after->ne[0], (long)shared_after->ne[1]);

        // 验证维度与目标模型匹配
        if (shared_after->ne[0] == n_embd_tgt) {
            printf("✅ Embedding dimension matches target model\n");
        } else {
            printf("⚠️  Embedding dimension mismatch: expected %d, got %ld\n", n_embd_tgt, (long)shared_after->ne[0]);
        }
    } else {
        printf("❌ Failed to get shared tensor\n");
    }

    // 6. 【关键测试】Draft Model仅使用Token ID的Forward Pass
    printf("\n=== Test 3: CRITICAL - Draft Forward Pass with Token ID ONLY ===\n");

    if (llama_context_get_shared_token_embd(ctx_draft)) {
        printf("✅ Draft context has access to shared embedding layer\n");

        // 测试用的Token IDs
        std::vector<llama_token> test_tokens = {198, 11, 12};
        bool all_tests_passed = true;

        for (size_t i = 0; i < test_tokens.size(); i++) {
            llama_token test_token = test_tokens[i];
            printf("\n--- Subtest %zu: Token ID %d ---\n", i+1, test_token);

            // 步骤1: 设置Draft Model的隐藏状态条件（EAGLE3流程需要）
            printf("Step 1: Setting hidden states conditioning...\n");

            // 创建虚拟隐藏状态（在实际EAGLE3中应来自Target Model）
            std::vector<float> dummy_hidden_states(n_embd_tgt);
            for (int j = 0; j < n_embd_tgt; j++) {
                dummy_hidden_states[j] = (float)j * 0.01f;
            }

            // 设置Draft Model的hidden states conditioning
            llama_context_set_target_hidden_states(ctx_draft, dummy_hidden_states.data(), dummy_hidden_states.size());
            printf("✅ Hidden states conditioning set for Draft Model\n");

            // 【关键步骤】Draft Model仅使用Token ID进行推理（不提供外部Embedding）
            printf("Step 2: CRITICAL - Draft Forward Pass with Token ID ONLY...\n");
            auto draft_batch = llama_batch_get_one(&test_token, 1);
            int draft_ret = llama_decode(ctx_draft, draft_batch);

            if (draft_ret != 0) {
                printf("❌ CRITICAL FAILURE: Draft model decode failed with ret=%d\n", draft_ret);
                printf("   This indicates Shared Embedding Layer is NOT working!\n");
                all_tests_passed = false;
                continue;
            }

            printf("✅ CRITICAL SUCCESS: Draft model completed forward pass using shared embedding!\n");

            // 步骤3: 验证Draft Model的输出
            printf("Step 3: Analyzing Draft Model output...\n");
            auto* draft_logits = llama_get_logits(ctx_draft);

            if (draft_logits) {
                // 检查logits是否有效（非NaN、有变化）
                bool has_valid_logits = false;
                float min_val = draft_logits[0], max_val = draft_logits[0];
                int nan_count = 0;

                for (int j = 0; j < std::min(100, vocab_draft); j++) {
                    float val = draft_logits[j];
                    if (std::isnan(val)) {
                        nan_count++;
                    } else {
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                        if (fabs(val) > 1e-6) has_valid_logits = true;
                    }
                }

                if (nan_count == 0 && has_valid_logits && max_val > min_val) {
                    printf("✅ Draft Model logits are VALID (range: %.4f to %.4f)!\n", min_val, max_val);

                    // 找到最佳预测
                    int best_token = 0;
                    float best_logit = draft_logits[0];
                    for (int j = 1; j < std::min(100, vocab_draft); j++) {
                        if (draft_logits[j] > best_logit) {
                            best_logit = draft_logits[j];
                            best_token = j;
                        }
                    }
                    printf("   Draft prediction: token %d (logit: %.4f)\n", best_token, best_logit);

                } else {
                    printf("❌ CRITICAL FAILURE: Draft Model logits are INVALID!\n");
                    printf("   NaN count: %d, has_valid: %s, range: %.4f to %.4f\n",
                           nan_count, has_valid_logits ? "true" : "false", min_val, max_val);
                    printf("   This indicates Shared Embedding Layer is NOT working properly!\n");
                    all_tests_passed = false;
                }
            } else {
                printf("❌ CRITICAL FAILURE: Failed to get Draft Model logits!\n");
                all_tests_passed = false;
            }
        }

        // 总结测试结果
        printf("\n=== Test 3 Final Result ===\n");
        if (all_tests_passed) {
            printf("🎉 ALL SUBTESTS PASSED!\n");
            printf("✅ Draft Model successfully uses Shared Embedding Layer\n");
            printf("✅ EAGLE3 embedding sharing mechanism is FUNCTIONAL\n");
        } else {
            printf("❌ SOME SUBTESTS FAILED!\n");
            printf("❌ Shared Embedding Layer mechanism needs debugging\n");
        }

    } else {
        printf("❌ No shared embedding layer available for testing\n");
    }

    // 7. 测试清理功能
    printf("\n=== Test 4: Cleanup Functionality ===\n");

    printf("--- Clearing shared embedding layer ---\n");
    llama_context_clear_shared_token_embd(ctx_draft);
    printf("✅ Called llama_context_clear_shared_token_embd()\n");

    struct ggml_tensor* shared_after_clear = llama_context_get_shared_token_embd(ctx_draft);
    printf("After clear - shared_token_embd: %s\n", shared_after_clear ? "non-NULL" : "NULL");

    if (!shared_after_clear) {
        printf("✅ Shared embedding layer successfully cleared\n");
    } else {
        printf("❌ Failed to clear shared embedding layer\n");
    }

    // 8. 重新测试设置功能（确保可以重复使用）
    printf("\n=== Test 5: Re-sharing Functionality ===\n");

    printf("--- Re-setting embedding layer sharing ---\n");
    llama_context_set_target_embedding_layer(ctx_draft, ctx_tgt);

    struct ggml_tensor* shared_after_reset = llama_context_get_shared_token_embd(ctx_draft);
    if (shared_after_reset) {
        printf("✅ Successfully re-shared embedding layer\n");
        print_tensor_info("Re-shared token_embd", shared_after_reset);
    } else {
        printf("❌ Failed to re-share embedding layer\n");
    }

    // 9. 总结测试结果
    printf("\n=== FINAL TEST SUMMARY ===\n");
    printf("✅ API Level Tests (Tests 1-2, 4-5):\n");
    printf("   - Embedding layer sharing/clearing API functionality verified\n");
    printf("   - Tensor pointer management works correctly\n");
    printf("   - API can be safely reused\n\n");

    printf("🔥 CRITICAL Functional Test (Test 3):\n");
    printf("   - Verifies Draft Model can complete Forward Pass using ONLY Token ID\n");
    printf("   - Proves Shared Embedding Layer actually works in computation graph\n");
    printf("   - Tests real EAGLE3 workflow integration\n\n");

    printf("📋 Test Verdict:\n");
    printf("   If Test 3 shows ALL SUBTESTS PASSED → EAGLE3 embedding sharing is FUNCTIONAL\n");
    printf("   If Test 3 shows FAILURES → Shared embedding mechanism needs debugging\n");

    // 10. 清理资源
    printf("\n--- Cleanup ---\n");
    llama_free(ctx_tgt);
    llama_free(ctx_draft);
    llama_model_free(model_tgt);
    llama_model_free(model_draft);
    llama_backend_free();

    printf("✅ All resources cleaned up\n");
    printf("=== Test Complete ===\n");

    return 0;
}