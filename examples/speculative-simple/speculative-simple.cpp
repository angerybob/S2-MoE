#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

struct moe_utility_cascade_state {
    bool enabled = false;
    int cap = 0;
    std::vector<int> ks;
    bool testing = true;
    int test_idx = 0;
    float test_sum = 0.0f;
    int test_n = 0;
    int test_iters = 4;
    int set_iters_base = 16;
    int set_left = 0;
    int set_mul = 1;
    int best_k = 0;
    float best_avg = -1.0e30f;
    int64_t pending_verify_us = 1;
    bool has_pending = false;

    void init(const common_params_speculative & sp) {
        enabled = sp.moe_utility_spec;
        if (!enabled) {
            return;
        }

        cap = std::max(0, sp.n_max);
        test_iters = std::max(1, sp.utility_test_iters);
        set_iters_base = std::max(1, sp.utility_set_iters);

        std::set<int> candidates;
        candidates.insert(0);
        if (cap >= 1) {
            candidates.insert(1);
        }
        if (cap >= 2) {
            candidates.insert(std::max(2, cap / 2));
        }
        candidates.insert(cap);
        ks.assign(candidates.begin(), candidates.end());

        testing = true;
        test_idx = 0;
        test_sum = 0.0f;
        test_n = 0;
        best_avg = -1.0e30f;
        best_k = cap;
        set_left = 0;
        set_mul = 1;
        has_pending = false;
    }

    int active_k() const {
        if (!enabled) {
            return cap;
        }
        if (testing) {
            return ks[std::min(test_idx, (int) ks.size() - 1)];
        }
        return best_k;
    }

    void on_verify_done(int64_t verify_us) {
        if (!enabled) {
            return;
        }
        pending_verify_us = std::max<int64_t>(1, verify_us);
        has_pending = true;
    }

    void on_outer_iter_start(int n_predict, int & n_predict_mark) {
        if (!enabled) {
            n_predict_mark = n_predict;
            return;
        }
        if (has_pending) {
            const int dtok = n_predict - n_predict_mark;
            const float util = (float) dtok / (float) pending_verify_us;
            has_pending = false;

            if (testing) {
                test_sum += util;
                test_n++;
                if (test_n >= test_iters) {
                    const float avg = test_sum / (float) test_n;
                    if (avg > best_avg) {
                        best_avg = avg;
                        best_k = ks[test_idx];
                    }
                    test_sum = 0.0f;
                    test_n = 0;
                    test_idx++;
                    if (test_idx >= (int) ks.size()) {
                        testing = false;
                        set_mul = best_k == 0 ? std::min(set_mul * 2, 8) : 1;
                        set_left = set_iters_base * set_mul;
                        test_idx = 0;
                        LOG_INF("%s: moe-utility-spec: test round done, best_k=%d best_avg_tokens_per_us=%e\n",
                                __func__, best_k, (double) best_avg);
                    }
                }
            } else {
                set_left--;
                if (set_left <= 0) {
                    testing = true;
                    test_idx = 0;
                    test_sum = 0.0f;
                    test_n = 0;
                    best_avg = -1.0e30f;
                    best_k = cap;
                    LOG_INF("%s: moe-utility-spec: restarting test phase (cap=%d)\n", __func__, cap);
                }
            }
        }
        n_predict_mark = n_predict;
    }
};

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    common_init();

    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    //llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;
    llama_context * ctx_encoder = NULL;
    llama_context * ctx_decoder = NULL;
    llama_context_ptr ctx_dft_shared_model;
    llama_model_ptr model_tgt_manual;
    llama_context_ptr ctx_tgt_manual;
    llama_model_ptr model_eagle3;
    llama_context_ptr ctx_eagle3_encoder;
    llama_context_ptr ctx_eagle3_decoder;

    int eagle3_tree_k = 1;
    if (params.speculative.eagle3) {
        if (const char * env = std::getenv("EAGLE3_TREE_K")) {
            eagle3_tree_k = std::max(1, std::atoi(env));
        }
    }

    common_params params_tgt = params;
    if (params.speculative.eagle3) {
        params_tgt.embedding = true;
        params_tgt.n_parallel = std::max(params_tgt.n_parallel, eagle3_tree_k);
    }

    common_params params_dft = params_tgt;
    params_dft.devices      = params.speculative.devices;
    params_dft.model        = params.speculative.model;
    params_dft.n_ctx        = params.speculative.n_ctx;
    params_dft.n_batch      = params.speculative.n_ctx > 0 ? params.speculative.n_ctx : params.n_batch;
    if (params.speculative.eagle3) {
        params_dft.n_ubatch = std::max(params_dft.n_ubatch, params_dft.n_batch);
    }
    params_dft.n_gpu_layers = params.speculative.n_gpu_layers;
    if (params.speculative.eagle3) {
        params_dft.n_expert_used = -1;
    } else if (params.speculative.n_expert_used > 0) {
        params_dft.n_expert_used = params.speculative.n_expert_used;
    }

    if (params.speculative.cpuparams.n_threads > 0) {
        params_dft.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params_dft.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    params_dft.tensor_buft_overrides     = params.speculative.tensor_buft_overrides;

    // load the target model
    common_init_result llama_init_tgt;
    if (params.speculative.eagle3) {
        auto mparams_dft = common_model_params_to_llama(params_dft);
        model_eagle3.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (!model_eagle3) {
            LOG_ERR("%s: failed to load EAGLE3 draft model '%s'\n", __func__, params_dft.model.path.c_str());
            return 1;
        }

        auto mparams_tgt = common_model_params_to_llama(params_tgt);
        model_tgt_manual.reset(llama_model_load_from_file(params_tgt.model.path.c_str(), mparams_tgt));
        if (!model_tgt_manual) {
            LOG_ERR("%s: failed to load target model '%s'\n", __func__, params_tgt.model.path.c_str());
            return 1;
        }

        auto cparams_tgt = common_context_params_to_llama(params_tgt);
        cparams_tgt.eagle3_model = model_eagle3.get();
        ctx_tgt_manual.reset(llama_init_from_model(model_tgt_manual.get(), cparams_tgt));
        if (!ctx_tgt_manual) {
            LOG_ERR("%s: failed to create target context with EAGLE3 extraction\n", __func__);
            return 1;
        }

        model_tgt = model_tgt_manual.get();
        ctx_tgt   = ctx_tgt_manual.get();
    } else {
        llama_init_tgt = common_init_from_params(params_tgt);
        model_tgt = llama_init_tgt.model.get();
        ctx_tgt   = llama_init_tgt.context.get();
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load or attach the draft model
    common_init_result llama_init_dft;
    const bool same_model_weights = params_dft.model.path == params_tgt.model.path &&
                                    params_dft.lora_adapters.empty() &&
                                    params_dft.control_vectors.empty();

    if (params.speculative.eagle3) {
        auto cparams_encoder = common_context_params_to_llama(params_dft);
        cparams_encoder.embeddings = true;
        ctx_eagle3_encoder.reset(llama_init_from_model(model_eagle3.get(), cparams_encoder));
        if (!ctx_eagle3_encoder) {
            LOG_ERR("%s: failed to create EAGLE3 encoder context\n", __func__);
            return 1;
        }

        auto cparams_decoder = common_context_params_to_llama(params_dft);
        cparams_decoder.embeddings = true;
        ctx_eagle3_decoder.reset(llama_init_from_model(model_eagle3.get(), cparams_decoder));
        if (!ctx_eagle3_decoder) {
            LOG_ERR("%s: failed to create EAGLE3 decoder context\n", __func__);
            return 1;
        }

        ctx_encoder = ctx_eagle3_encoder.get();
        ctx_decoder = ctx_eagle3_decoder.get();
        ctx_dft = ctx_decoder;

        llama_context_set_draft_context(ctx_decoder, true);
        llama_context_set_target_embedding_layer(ctx_decoder, ctx_tgt);

        LOG_INF("%s: EAGLE3 speculative decoding enabled; using encoder and decoder contexts\n", __func__);
    } else if (same_model_weights) {
        auto cparams_dft = common_context_params_to_llama(params_dft);
        ctx_dft_shared_model.reset(llama_init_from_model(model_tgt, cparams_dft));
        if (!ctx_dft_shared_model) {
            LOG_ERR("%s: failed to create draft context from shared target model '%s'\n", __func__, params_tgt.model.path.c_str());
            return 1;
        }
        ctx_dft = ctx_dft_shared_model.get();
        if (params_dft.warmup) {
            const llama_vocab * vocab_dft = llama_model_get_vocab(model_tgt);
            std::vector<llama_token> tmp;
            const llama_token bos = llama_vocab_bos(vocab_dft);
            const llama_token eos = llama_vocab_eos(vocab_dft);
            if (bos != LLAMA_TOKEN_NULL) {
                tmp.push_back(bos);
            }
            if (eos != LLAMA_TOKEN_NULL) {
                tmp.push_back(eos);
            }
            if (tmp.empty()) {
                tmp.push_back(0);
            }
            llama_set_warmup(ctx_dft, true);
            llama_decode(ctx_dft, llama_batch_get_one(tmp.data(), std::min(tmp.size(), (size_t) params_dft.n_batch)));
            llama_memory_clear(llama_get_memory(ctx_dft), true);
            llama_synchronize(ctx_dft);
            llama_set_warmup(ctx_dft, false);
        }
        LOG_INF("%s: draft and target use the same model weights; created a separate draft context only\n", __func__);
    } else {
        llama_init_dft = common_init_from_params(params_dft);
        ctx_dft = llama_init_dft.context.get();
    }

    //model_dft = llama_init_dft.model.get();
    if (ctx_dft == nullptr) {
        LOG_ERR("%s: failed to initialize draft context\n", __func__);
        return 1;
    }

    if (!params.speculative.eagle3 && !common_speculative_are_compatible(ctx_tgt, ctx_dft)) {
        LOG_INF("the draft model '%s' is not compatible with the target model '%s'. tokens will be translated between the draft and target models.\n", params_dft.model.path.c_str(), params_tgt.model.path.c_str());
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    // how many tokens to draft each time
    int n_draft     = params.speculative.n_max;
    int n_draft_min = params.speculative.n_min;

    moe_utility_cascade_state cascade;
    cascade.init(params.speculative);
    if (cascade.enabled) {
        LOG_INF("%s: MoE utility-driven speculation enabled: cap=%d test_iters=%d set_iters=%d\n",
                __func__, n_draft, params.speculative.utility_test_iters, params.speculative.utility_set_iters);
    }

    float p_min = params.speculative.p_min;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    int n_tree_steps = 0;
    int n_tree_hits  = 0;

    // used to determine end of generation
    bool has_eos = false;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    llama_token id_last;
    llama_tokens prompt_tgt;
    int n_past;

    if (params.speculative.eagle3) {
        llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size()));
        id_last = common_sampler_sample(smpl, ctx_tgt, -1);
        common_sampler_accept(smpl, id_last, true);

        for (int32_t seq = 1; seq < eagle3_tree_k; ++seq) {
            llama_memory_seq_cp(llama_get_memory(ctx_tgt), 0, seq, -1, -1);
        }

        const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
        LOG("%s", token_str.c_str());

        prompt_tgt.assign(inp.begin(), inp.end());
        n_past = inp.size();
        n_predict++;
    } else {
        // eval the prompt
        llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

        // note: keep the last token separate!
        id_last = inp.back();

        // all tokens currently in the target context
        prompt_tgt.assign(inp.begin(), inp.end() - 1);
        n_past = inp.size() - 1;
    }
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    // init the speculator
    struct common_speculative_params params_spec;
    params_spec.n_draft = n_draft;
    params_spec.n_reuse = llama_n_ctx(params.speculative.eagle3 ? ctx_decoder : ctx_dft) - n_draft;
    params_spec.p_min   = p_min;

    struct common_speculative * spec = params.speculative.eagle3 ?
        common_speculative_init_eagle3(ctx_tgt, ctx_encoder, ctx_decoder) :
        common_speculative_init(ctx_tgt, ctx_dft);
    for (auto &pair : params.speculative.replacements) {
        common_speculative_add_replacement_tgt_dft(spec, pair.first.c_str(), pair.second.c_str());
    }

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, std::max(1, eagle3_tree_k));

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    int n_predict_mark = n_predict;

    while (true) {
        cascade.on_outer_iter_start(n_predict, n_predict_mark);
        const int n_draft_active = cascade.enabled ? cascade.active_k() : n_draft;
        params_spec.n_draft = n_draft_active;
        params_spec.n_reuse = llama_n_ctx(params.speculative.eagle3 ? ctx_decoder : ctx_dft) - n_draft_active;

        if (params.speculative.eagle3 && eagle3_tree_k > 1) {
            const llama_tokens candidates =
                common_speculative_gen_eagle3_topk(spec, prompt_tgt, id_last, eagle3_tree_k);

            common_batch_clear(batch_tgt);
            for (int32_t lane = 0; lane < (int32_t) candidates.size(); ++lane) {
                common_batch_add(batch_tgt, id_last,          n_past,     { lane }, true);
                common_batch_add(batch_tgt, candidates[lane], n_past + 1, { lane }, true);
            }

            const int64_t t_verify_start = ggml_time_us();
            GGML_ASSERT(llama_decode(ctx_tgt, batch_tgt) == 0);
            const int64_t t_verify_end = ggml_time_us();
            cascade.on_verify_done(t_verify_end - t_verify_start);

            llama_tokens ids;
            const llama_token sampled = common_sampler_sample(smpl, ctx_tgt, 0);
            common_sampler_accept(smpl, sampled, true);
            ids.push_back(sampled);

            int32_t selected_lane = 0;
            bool tree_hit = false;
            for (int32_t lane = 0; lane < (int32_t) candidates.size(); ++lane) {
                if (candidates[lane] == sampled) {
                    selected_lane = lane;
                    tree_hit = true;
                    const llama_token next = common_sampler_sample(smpl, ctx_tgt, 2 * lane + 1);
                    common_sampler_accept(smpl, next, true);
                    ids.push_back(next);
                    break;
                }
            }

            const int32_t feature_indices[2] = { 2 * selected_lane, 2 * selected_lane + 1 };
            llama_select_eagle3_target_features(ctx_tgt, feature_indices, tree_hit ? 2 : 1);

            n_past += tree_hit ? 2 : 1;
            auto * mem_tgt = llama_get_memory(ctx_tgt);
            for (int32_t lane = 0; lane < eagle3_tree_k; ++lane) {
                if (lane == selected_lane) {
                    continue;
                }
                llama_memory_seq_rm(mem_tgt, lane, -1, -1);
                llama_memory_seq_cp(mem_tgt, selected_lane, lane, -1, -1);
            }
            llama_memory_seq_rm(mem_tgt, -1, n_past, -1);

            n_drafted += candidates.size();
            n_accept  += tree_hit ? 1 : 0;
            n_tree_steps++;
            n_tree_hits += tree_hit ? 1 : 0;
            n_predict += ids.size();

            for (size_t i = 0; i < ids.size(); ++i) {
                prompt_tgt.push_back(id_last);
                id_last = ids[i];

                if (llama_vocab_is_eog(vocab, id_last)) {
                    has_eos = true;
                    break;
                }

                const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
                if (params.use_color && i + 1 < ids.size()) {
                    LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
                } else {
                    LOG("%s", token_str.c_str());
                }
            }

            if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
                break;
            }
            continue;
        }

        // optionally, generate draft tokens that can be appended to the target batch
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        llama_tokens draft = common_speculative_gen_draft(spec, params_spec, prompt_tgt, id_last);

        //LOG_DBG("draft: %s\n", string_from(ctx_dft, draft).c_str());

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            // do not waste time on small drafts
            if (draft.size() < (size_t) n_draft_min) {
                draft.clear();
            }

            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            const int64_t t_verify_start = ggml_time_us();
            llama_decode(ctx_tgt, batch_tgt);
            const int64_t t_verify_end = ggml_time_us();
            cascade.on_verify_done(t_verify_end - t_verify_start);
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        const auto ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        n_past    += ids.size() - 1;
        n_drafted += draft.size(); // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d (cap; moe-utility-spec %s)\n", n_draft, cascade.enabled ? "on" : "off");
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);
    if (n_tree_steps > 0) {
        LOG_INF("tree_hit  = %d / %d (%.3f%%)\n",
                n_tree_hits, n_tree_steps, 100.0f * n_tree_hits / n_tree_steps);
    }

    LOG_INF("\n");
    if (params.speculative.eagle3) {
        LOG_INF("EAGLE3 encoder:\n\n");
        llama_perf_context_print(ctx_encoder);

        LOG_INF("\n");
        LOG_INF("EAGLE3 decoder:\n\n");
        llama_perf_context_print(ctx_decoder);
    } else {
        LOG_INF("draft:\n\n");
        llama_perf_context_print(ctx_dft);
    }

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
