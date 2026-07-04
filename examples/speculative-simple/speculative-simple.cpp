#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"
#include "chat.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

// MoE utility-driven draft length (Cascade-style: periodic test/set, arXiv:2506.20675)
struct MoeUtilityCascadeState {
    bool enabled = false;
    int cap = 0;
    std::vector<int> ks;
    bool testing = true;
    int test_idx = 0;
    float test_sum = 0.f;
    int test_n = 0;
    int test_iters = 4;
    int set_iters_base = 16;
    int set_left = 0;
    int set_mul = 1;
    int best_k = 0;
    float best_avg = -1e30f;
    int64_t pending_v_us = 1;
    bool has_pending = false;

    void init(const common_params_speculative & sp) {
        enabled = sp.moe_utility_spec;
        if (!enabled) {
            return;
        }
        cap = std::max(0, (int) sp.n_max);
        test_iters = std::max(1, (int) sp.utility_test_iters);
        set_iters_base = std::max(1, (int) sp.utility_set_iters);
        std::set<int> cand;
        cand.insert(0);
        if (cap >= 1) {
            cand.insert(1);
        }
        if (cap >= 2) {
            const int mid = std::max(2, cap / 2);
            if (mid <= cap) {
                cand.insert(mid);
            }
        }
        cand.insert(cap);
        ks.assign(cand.begin(), cand.end());
        testing = true;
        test_idx = 0;
        test_sum = 0.f;
        test_n = 0;
        best_avg = -1e30f;
        best_k = cap;
        has_pending = false;
        set_mul = 1;
        set_left = 0;
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

    void on_verify_done(int64_t v_us) {
        if (!enabled) {
            return;
        }
        pending_v_us = std::max<int64_t>(1, v_us);
        has_pending = true;
    }

    void on_outer_iter_start(int n_predict, int & n_predict_mark) {
        if (!enabled) {
            n_predict_mark = n_predict;
            return;
        }
        if (has_pending) {
            const int dtok = n_predict - n_predict_mark;
            const float util = float(dtok) / float(pending_v_us);
            has_pending = false;

            if (testing) {
                test_sum += util;
                test_n++;
                if (test_n >= test_iters) {
                    const float avg = test_sum / float(test_n);
                    if (avg > best_avg) {
                        best_avg = avg;
                        best_k = ks[test_idx];
                    }
                    test_sum = 0.f;
                    test_n = 0;
                    test_idx++;
                    if (test_idx >= (int) ks.size()) {
                        testing = false;
                        set_mul = (best_k == 0) ? std::min(set_mul * 2, 8) : 1;
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
                    test_sum = 0.f;
                    test_n = 0;
                    best_avg = -1e30f;
                    best_k = cap;
                    LOG_INF("%s: moe-utility-spec: restarting test phase (cap=%d)\n", __func__, cap);
                }
            }
        }
        n_predict_mark = n_predict;
    }
};

// Simple JSON string extractor: supports "key": "value" and "key": ["value", ...] (returns first string).
static std::string extract_json_value(const std::string & json_line, const std::string & key) {
    const std::string search_key = "\"" + key + "\"";
    const size_t key_pos = json_line.find(search_key);
    if (key_pos == std::string::npos) {
        return "";
    }

    const size_t colon_pos = json_line.find(':', key_pos + search_key.length());
    if (colon_pos == std::string::npos) {
        return "";
    }

    const size_t start_quote = json_line.find('"', colon_pos + 1);
    if (start_quote == std::string::npos) {
        return "";
    }

    std::string result;
    bool escape = false;
    for (size_t i = start_quote + 1; i < json_line.length(); ++i) {
        const char c = json_line[i];
        if (escape) {
            switch (c) {
                case 'n': result += '\n'; break;
                case 'r': result += '\r'; break;
                case 't': result += '\t'; break;
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                default: result += c; break;
            }
            escape = false;
        } else {
            if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                return result;
            } else {
                result += c;
            }
        }
    }
    return "";
}

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

    if (params.speculative.moe_utility_spec) {
        LOG_INF("%s: MoE utility-driven speculation (Cascade-style): --draft cap=%d test_iters=%d set_iters=%d\n",
                __func__, params.speculative.n_max, params.speculative.utility_test_iters, params.speculative.utility_set_iters);
    }

    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // EAGLE3 specific contexts
    llama_context * ctx_encoder = NULL;
    llama_context * ctx_decoder = NULL;

    common_init_result init_tgt;
    common_init_result init_dft;

    const bool native_draft = params.speculative.eagle3 || params.speculative.dflash;

    // Native EAGLE3/DFlash paths load the draft model first so the target
    // context can enable its requested feature extraction layers.
    if (native_draft) {
        llama_model_params dft_mp = llama_model_default_params();
        dft_mp.n_gpu_layers = params.speculative.n_gpu_layers;
        // Draft models must never enter the target SSD expert registry.
        dft_mp.use_ssd_moe = false;
        dft_mp.hot_experts_path = nullptr;
        dft_mp.use_mmap = params.use_mmap;
        model_dft = llama_model_load_from_file(params.speculative.model.path.c_str(), dft_mp);
        if (!model_dft) {
            LOG_ERR("failed to load native draft model\n");
            return 1;
        }

        llama_model_params tgt_mp = llama_model_default_params();
        tgt_mp.n_gpu_layers = params.n_gpu_layers;
        tgt_mp.use_ssd_moe = params.use_ssd_moe;
        tgt_mp.hot_experts_path = params.hot_experts_path.empty() ? nullptr : params.hot_experts_path.c_str();
        tgt_mp.use_mmap = params.use_mmap;
        model_tgt = llama_model_load_from_file(params.model.path.c_str(), tgt_mp);
        if (!model_tgt) {
            LOG_ERR("failed to load target model\n");
            return 1;
        }

        llama_context_params tcp = common_context_params_to_llama(params);
        tcp.eagle3_model = model_dft;  // Enable feature extraction
        ctx_tgt = llama_init_from_model(model_tgt, tcp);
        if (!ctx_tgt) {
            LOG_ERR("failed to create target context with draft feature extraction\n");
            return 1;
        }
    } else {
        // Standard load the target model
        init_tgt = common_init_from_params(params);
        model_tgt = init_tgt.model.get();
        ctx_tgt   = init_tgt.context.get();
    }

    // force a known chat template unless user already specified one
    if (params.chat_template.empty() || params.chat_template == "auto") {
        params.chat_template = "chatml";
    }
    params.enable_chat_template = true;

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);
    auto chat_templates = common_chat_templates_init(model_tgt, params.chat_template);
    const bool has_chat_template = common_chat_templates_was_explicit(chat_templates.get());

    // load the draft model
    params.devices      = params.speculative.devices;
    params.model        = params.speculative.model;
    params.n_ctx        = params.speculative.n_ctx;
    params.n_batch      = params.speculative.n_ctx > 0 ? params.speculative.n_ctx : params.n_batch;
    params.n_gpu_layers = params.speculative.n_gpu_layers;

    if (params.speculative.cpuparams.n_threads > 0) {
        params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    params.tensor_buft_overrides     = params.speculative.tensor_buft_overrides;

    if (params.speculative.eagle3) {
        // EAGLE3: create encoder and decoder contexts
        llama_context_params enc_params = common_context_params_to_llama(params);
        enc_params.embeddings = true;
        ctx_encoder = llama_init_from_model(model_dft, enc_params);
        if (!ctx_encoder) {
            LOG_ERR("failed to create EAGLE3 encoder context\n");
            return 1;
        }

        llama_context_params dec_params = common_context_params_to_llama(params);
        dec_params.target_model = model_tgt;
        dec_params.embeddings = true;
        ctx_decoder = llama_init_from_model(model_dft, dec_params);
        if (!ctx_decoder) {
            LOG_ERR("failed to create EAGLE3 decoder context\n");
            return 1;
        }
    } else if (params.speculative.dflash) {
        const int32_t block_size = llama_model_dflash_block_size(model_dft);
        if (block_size < 2) {
            LOG_ERR("draft model is not a valid DFlash model\n");
            return 1;
        }
        if (params.speculative.n_max > block_size - 1) {
            LOG_WRN("requested %d draft tokens exceeds DFlash block capacity; clamping to %d\n",
                    params.speculative.n_max, block_size - 1);
            params.speculative.n_max = block_size - 1;
        }

        llama_context_params dflash_params = common_context_params_to_llama(params);
        dflash_params.target_model = model_tgt;
        dflash_params.embeddings = true;
        dflash_params.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
        ctx_dft = llama_init_from_model(model_dft, dflash_params);
        if (!ctx_dft) {
            LOG_ERR("failed to create DFlash context\n");
            return 1;
        }
    } else {
        // Standard: load draft model context
        init_dft = common_init_from_params(params);
        model_dft = init_dft.model.get();
        ctx_dft   = init_dft.context.get();

        if (!common_speculative_are_compatible(ctx_tgt, ctx_dft)) {
            LOG_INF("the draft model '%s' is not compatible with the target model '%s'. tokens will be translated between the draft and target models.\n", params.speculative.model.path.c_str(), params.model.path.c_str());
        }
    }

    const bool use_dataset = !params.dataset_path.empty();
    std::ifstream dataset_file;
    if (use_dataset) {
        dataset_file.open(params.dataset_path);
        if (!dataset_file.is_open()) {
            LOG_ERR("%s: failed to open dataset file: %s\n", __func__, params.dataset_path.c_str());
            return 1;
        }
    } else if (params.prompt.empty()) {
        LOG_ERR("%s: no prompt provided\n", __func__);
        return 1;
    }

    double total_t_enc_us = 0.0;
    double total_t_dec_us = 0.0;
    long long total_n_input = 0;
    long long total_n_predict = 0;
    long long total_n_drafted = 0;
    long long total_n_accept = 0;
    common_speculative_acceptance_metrics total_acceptance_metrics;
    int processed_count = 0;
    int question_idx = 0;

    std::string json_line;
    while (true) {
        std::string question;
        if (use_dataset) {
            if (!std::getline(dataset_file, json_line)) {
                break;
            }
            if (json_line.empty()) {
                continue;
            }
            if (params.n_questions_limit > 0 && processed_count >= params.n_questions_limit) {
                LOG_INF("Reached question limit (%d). Stopping.\n", params.n_questions_limit);
                break;
            }
            question = extract_json_value(json_line, "question");
            if (question.empty()) {
                question = extract_json_value(json_line, "prompt");
            }
            if (question.empty()) {
                question = extract_json_value(json_line, "turns");
            }
            if (question.empty()) {
                LOG_WRN("Skipping line %d: could not find 'question', 'prompt' or 'turns' key.\n", question_idx + 1);
                continue;
            }
            ++question_idx;
        } else {
            if (processed_count >= 1) {
                break;
            }
            question = params.prompt;
            question_idx = 1;
        }

        std::string prompt = question;
        if (has_chat_template) {
            std::vector<common_chat_msg> chat_msgs;
            if (!params.system_prompt.empty()) {
                chat_msgs.push_back({"system", params.system_prompt});
            } else {
                chat_msgs.push_back({
                    "system",
                    "You are a helpful assistant. "
                    "Do NOT output analysis, reasoning, or chain-of-thought. "
                });
            }
            chat_msgs.push_back({"user", question});

            common_chat_templates_inputs inputs;
            inputs.messages = chat_msgs;
            inputs.add_generation_prompt = true;
            inputs.use_jinja = params.use_jinja;
            inputs.parallel_tool_calls = false;

            prompt = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            if (native_draft) {
                LOG_INF("%s: native draft chat template applied\n", __func__);
            }
        }

        // Tokenize the prompt
        std::vector<llama_token> inp = common_tokenize(ctx_tgt, prompt, true, true);

        if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
            LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));
            continue;
        }

        if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
            LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));
            continue;
        }
        if (params.speculative.eagle3 && ctx_encoder && llama_n_ubatch(ctx_encoder) < (uint32_t) inp.size()) {
            LOG_ERR("%s: EAGLE3 encoder requires n_ubatch >= prompt tokens (%d > %d). Use --ubatch-size to increase.\n",
                    __func__, (int) inp.size(), llama_n_ubatch(ctx_encoder));
            continue;
        }

        LOG("\n\n=== Processing Question %d ===\n", question_idx);
        for (auto id : inp) {
            LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
        }

        // reset KV cache per question (seq 0 only) to avoid full clear crashes
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);
        if (ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, 0, -1);
        }
        if (ctx_encoder) {
            llama_memory_seq_rm(llama_get_memory(ctx_encoder), 0, 0, -1);
        }
        if (ctx_decoder) {
            llama_memory_seq_rm(llama_get_memory(ctx_decoder), 0, 0, -1);
        }

        const int n_draft_cap = params.speculative.n_max;
        const int n_draft_min = params.speculative.n_min;
        const float p_min = params.speculative.p_min;

        int n_predict = 0;
        int n_drafted = 0;
        int n_accept  = 0;
        common_speculative_acceptance_metrics acceptance_metrics;

        // used to determine end of generation
        bool has_eos = false;

        const auto t_enc_start = ggml_time_us();

        // target model sampling context
        struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

        // eval the prompt
        llama_token id_last;
        llama_tokens prompt_tgt;
        int n_past;

        if (params.speculative.eagle3) {
            // Target model decodes full prompt and sample first token and intermediate features are extracted
            llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size()));

            id_last = common_sampler_sample(smpl, ctx_tgt, -1);
            common_sampler_accept(smpl, id_last, true);
            LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());
            n_predict++;

            // all tokens currently in the target context
            prompt_tgt.assign(inp.begin(), inp.end());
            prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

            n_past = inp.size();
        } else if (params.speculative.dflash) {
            // DFlash also needs target features for every prompt position.
            llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size()));

            id_last = common_sampler_sample(smpl, ctx_tgt, -1);
            common_sampler_accept(smpl, id_last, true);
            LOG("%s", common_token_to_piece(ctx_tgt, id_last).c_str());
            n_predict++;

            prompt_tgt.assign(inp.begin(), inp.end());
            prompt_tgt.reserve(llama_n_ctx(ctx_tgt));
            n_past = inp.size();
        } else {
            llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

            // note: keep the last token separate!
            id_last = inp.back();

            // all tokens currently in the target context
            prompt_tgt.assign(inp.begin(), inp.end() - 1);
            prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

            n_past = inp.size() - 1;
        }

        // init the speculator
        struct common_speculative_params params_spec;
        params_spec.p_min = p_min;

        struct common_speculative * spec = NULL;

        if (params.speculative.eagle3) {
            spec = common_speculative_init_eagle3(ctx_tgt, ctx_encoder, ctx_decoder);
        } else if (params.speculative.dflash) {
            spec = common_speculative_init_dflash(ctx_tgt, ctx_dft);
        } else {
            params_spec.n_reuse = llama_n_ctx(ctx_dft) - n_draft_cap;
            spec = common_speculative_init(ctx_tgt, ctx_dft);
            for (auto &pair : params.speculative.replacements) {
                common_speculative_add_replacement_tgt_dft(spec, pair.first.c_str(), pair.second.c_str());
            }
        }

        llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

        MoeUtilityCascadeState cascade;
        cascade.init(params.speculative);
        int n_predict_mark = 0;

        const auto t_enc_end = ggml_time_us();

        const auto t_dec_start = ggml_time_us();

        while (true) {
            cascade.on_outer_iter_start(n_predict, n_predict_mark);
            const int n_draft_active = cascade.enabled ? cascade.active_k() : n_draft_cap;
            params_spec.n_draft = n_draft_active;
            if (!native_draft && ctx_dft) {
                params_spec.n_reuse = llama_n_ctx(ctx_dft) - n_draft_active;
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

            const auto t_verify_start = ggml_time_us();
            llama_decode(ctx_tgt, batch_tgt);
            const auto t_verify_end = ggml_time_us();
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
        acceptance_metrics.add(ids.size() - 1, draft.size());
        total_acceptance_metrics.add(ids.size() - 1, draft.size());

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
        LOG_INF("n_draft   = %d (cap; moe-utility-spec %s)\n", n_draft_cap, cascade.enabled ? "on" : "off");
        LOG_INF("n_predict = %d\n", n_predict);
        LOG_INF("n_drafted = %d\n", n_drafted);
        LOG_INF("n_accept  = %d\n", n_accept);
        LOG_INF("accept    = %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);
        LOG_INF("mean accepted draft length = %.3f\n", acceptance_metrics.mean_accepted_length());
        LOG_INF("first draft position acceptance = %.3f%%\n",
                100.0 * acceptance_metrics.position_acceptance(0));

        LOG_INF("\n");
        LOG_INF("draft:\n\n");

        if (ctx_dft) {
            llama_perf_context_print(ctx_dft);
        } else if (ctx_encoder && ctx_decoder) {
            LOG_INF(" Eagle3 Draft encoder:\n");
            llama_perf_context_print(ctx_encoder);
            LOG_INF("\nEagle3 Draft decoder:\n");
            llama_perf_context_print(ctx_decoder);
        }

        LOG_INF("\n");
        LOG_INF("target:\n\n");
        common_perf_print(ctx_tgt, smpl);

        llama_batch_free(batch_tgt);
        common_sampler_free(smpl);
        common_speculative_free(spec);

        total_t_enc_us += (t_enc_end - t_enc_start);
        total_t_dec_us += (t_dec_end - t_dec_start);
        total_n_input += n_input;
        total_n_predict += n_predict;
        total_n_drafted += n_drafted;
        total_n_accept += n_accept;
        processed_count++;
    }


    LOG("\n\n==================== FINAL STATISTICS ====================\n");
    LOG("Total Questions Processed: %d\n", processed_count);
    if (processed_count > 0) {
        const double total_enc_sec = total_t_enc_us / 1e6;
        const double total_dec_sec = total_t_dec_us / 1e6;
        const double avg_enc_speed = total_enc_sec > 0.0 ? (total_n_input / total_enc_sec) : 0.0;
        const double avg_dec_speed = total_dec_sec > 0.0 ? (total_n_predict / total_dec_sec) : 0.0;
        const double avg_accept_rate = total_n_drafted > 0 ? (100.0 * total_n_accept / total_n_drafted) : 0.0;
        const double overall_acceptance_length = (total_n_predict - total_n_accept) > 0 ? 
                ((double)total_n_accept / (double)(total_n_predict - total_n_accept)) : 0.0;
        LOG_INF("Total Input Tokens:    %lld\n", total_n_input);
        LOG_INF("Total Gen Tokens:      %lld\n", total_n_predict);
        LOG_INF("Total Encoding Time:   %.3f s\n", total_enc_sec);
        LOG_INF("Total Decoding Time:   %.3f s\n", total_dec_sec);
        LOG("\n");
        LOG_INF("Avg Encoding Speed:    %.3f t/s\n", avg_enc_speed);
        LOG_INF("Avg Decoding Speed:    %.3f t/s\n", avg_dec_speed);
        LOG("\n");
        LOG_INF("Total Drafted:         %lld\n", total_n_drafted);
        LOG_INF("Total Accepted:        %lld\n", total_n_accept);
        LOG_INF("Overall Accept Rate:   %.3f%%\n", avg_accept_rate);
        LOG_INF("Overall Acceptance Length: %.3f drafted tokens accepted per non-accepted token\n", overall_acceptance_length);
        LOG_INF("Mean Accepted Draft Length: %.3f\n", total_acceptance_metrics.mean_accepted_length());
        LOG_INF("First Draft Position Acceptance: %.3f%%\n",
                100.0 * total_acceptance_metrics.position_acceptance(0));
    }
    LOG("==========================================================\n\n");

    if (native_draft) {
        if (ctx_decoder) {
            llama_free(ctx_decoder);
        }
        if (ctx_encoder) {
            llama_free(ctx_encoder);
        }
        if (ctx_dft) {
            llama_free(ctx_dft);
        }
        llama_free(ctx_tgt);
        llama_model_free(model_tgt);
        llama_model_free(model_dft);
    }

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
