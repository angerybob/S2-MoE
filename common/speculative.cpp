#include "speculative.h"

#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "common.h"
#include "sampling.h"

#include <cstring>
#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct common_speculative {
    struct llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    struct llama_context * ctx_dft;
    struct common_sampler * smpl;

    llama_batch batch;
    llama_tokens prompt_dft;
    bool vocab_dft_compatible = true; // whether retokenization is needed
    std::map<std::string, std::string> tgt_dft_replacements = {};

    struct llama_context * eagle3_encoder = nullptr;
    struct llama_context * eagle3_decoder = nullptr;
    int32_t eagle3_n_past = 0;

    bool dflash = false;
    llama_batch dflash_batch_inject = {};
    int32_t dflash_n_past = 0;
    int32_t dflash_block_size = 0;
    llama_token dflash_mask_token = -1;
    int32_t dflash_n_embd_enc = 0;
    int32_t dflash_n_embd_dec = 0;
};

struct common_speculative * common_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft) {
    auto * result = new common_speculative {
        /* .ctx_tgt    = */ ctx_tgt,
        /* .ctx_dft    = */ ctx_dft,
        /* .smpl       = */ nullptr,
        /* .batch      = */ llama_batch_init(llama_n_batch(ctx_dft), 0, 1),
        /* .prompt_dft = */ {},
        /* .vocab_dft_compatible = */ false,
    };

    // TODO: optimize or pass from outside?
#if 0
    {
        common_params_sampling params;
        params.no_perf = false;

        params.top_k = 40;
        params.top_p = 0.9;

        params.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
            COMMON_SAMPLER_TYPE_TOP_P,
            COMMON_SAMPLER_TYPE_INFILL,
        };

        result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
    }
#else
    {
        common_params_sampling params;
        params.no_perf = false;

        params.top_k = 10;

        params.samplers = {
            COMMON_SAMPLER_TYPE_TOP_K,
        };

        result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
    }
#endif

    result->vocab_dft_compatible = common_speculative_are_compatible(ctx_tgt, ctx_dft);
    LOG_DBG("vocab_dft_compatible = %d\n", result->vocab_dft_compatible);

    return result;
}

struct common_speculative * common_speculative_init_eagle3(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_encoder,
        struct llama_context * ctx_decoder) {
    auto * result = new common_speculative {
        /* .ctx_tgt    = */ ctx_tgt,
        /* .ctx_dft    = */ nullptr,
        /* .smpl       = */ nullptr,
        /* .batch      = */ llama_batch_init(llama_n_batch(ctx_decoder), 0, 1),
        /* .prompt_dft = */ {},
        /* .vocab_dft_compatible = */ true,
        /* .tgt_dft_replacements = */ {},
        /* .eagle3_encoder = */ ctx_encoder,
        /* .eagle3_decoder = */ ctx_decoder,
        /* .eagle3_n_past = */ 0,
    };

    common_params_sampling params;
    params.no_perf = false;
    params.top_k = 10;
    params.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
    result->smpl = common_sampler_init(llama_get_model(ctx_decoder), params);

    return result;
}

struct common_speculative * common_speculative_init_dflash(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft) {
    auto * result = common_speculative_init(ctx_tgt, ctx_dft);
    const llama_model * model_dft = llama_get_model(ctx_dft);
    const llama_model * model_tgt = llama_get_model(ctx_tgt);

    result->dflash = true;
    result->dflash_block_size = llama_model_dflash_block_size(model_dft);
    result->dflash_mask_token = llama_model_dflash_mask_token(model_dft);
    result->dflash_n_embd_dec = llama_model_n_embd(model_dft);
    result->dflash_n_embd_enc =
        llama_model_n_embd(model_tgt) *
        static_cast<int32_t>(llama_model_dflash_target_layer_count(model_dft));

    GGML_ASSERT(result->dflash_block_size >= 2);
    GGML_ASSERT(result->dflash_mask_token >= 0);
    GGML_ASSERT(result->dflash_n_embd_enc > 0);

    result->dflash_batch_inject = llama_batch_init(
        llama_n_batch(ctx_dft),
        result->dflash_n_embd_dec,
        1);

    common_sampler_free(result->smpl);
    common_params_sampling params;
    params.no_perf = false;
    params.top_k = 1;
    params.samplers = {COMMON_SAMPLER_TYPE_TOP_K};
    result->smpl = common_sampler_init(model_dft, params);

    return result;
}

void common_speculative_free(struct common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    common_sampler_free(spec->smpl);

    llama_batch_free(spec->batch);
    if (spec->dflash) {
        llama_batch_free(spec->dflash_batch_inject);
    }

    delete spec;
}

bool common_speculative_are_compatible(
    const struct llama_context * ctx_tgt,
    const struct llama_context * ctx_dft) {
    const struct llama_model * model_tgt = llama_get_model(ctx_tgt);
    const struct llama_model * model_dft = llama_get_model(ctx_dft);

    const struct llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const struct llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

void common_speculative_add_replacement_tgt_dft(
        struct common_speculative * spec,
        const char *source, const char *dest) {
    spec->tgt_dft_replacements[source] = dest;
}

static std::string replace_to_dft(
        struct common_speculative * spec,
        const std::string& input) {
    std::string result = input;
    for (const auto & pair : spec->tgt_dft_replacements) {
        size_t pos = result.find(pair.first);
        while (pos != std::string::npos) {
            result.replace(pos, pair.first.length(), pair.second);
            pos = result.find(pair.first, pos + pair.second.length());
        }
    }
    return result;
}

static std::string replace_to_tgt(
        struct common_speculative * spec,
        const std::string& input) {
    std::string result = input;
    for (const auto& pair : spec->tgt_dft_replacements) {
        size_t pos = result.find(pair.second);
        while (pos != std::string::npos) {
            result.replace(pos, pair.second.length(), pair.first);
            pos = result.find(pair.second, pos + pair.first.length());
        }
    }
    return result;
}

static llama_tokens common_speculative_gen_eagle3_draft(
        struct common_speculative * spec,
        struct common_speculative_params params,
        const llama_tokens & prompt_tgt,
        llama_token id_last) {
    auto * ctx_tgt     = spec->ctx_tgt;
    auto * ctx_encoder = spec->eagle3_encoder;
    auto * ctx_decoder = spec->eagle3_decoder;
    auto * smpl        = spec->smpl;
    auto & batch       = spec->batch;

    const int n_embd = llama_model_n_embd(llama_get_model(ctx_encoder));
    const int n = (int) prompt_tgt.size();
    const int n_new = n - spec->eagle3_n_past;

    GGML_ASSERT(n >= 1);
    GGML_ASSERT(n_new >= 1);

    llama_memory_seq_rm(llama_get_memory(ctx_decoder), 0, spec->eagle3_n_past, -1);

    const float * features = llama_get_eagle3_target_features(ctx_tgt);
    GGML_ASSERT(features != nullptr && "no EAGLE3 target features");
    const size_t feature_size = llama_get_eagle3_target_features_size(ctx_tgt);
    const size_t feature_row_size = (size_t) n_embd * 3;
    GGML_ASSERT(feature_size >= feature_row_size * (size_t) n_new &&
            "not enough EAGLE3 target feature rows for draft encoder");

    llama_batch enc_batch = {
        /*.n_tokens  =*/ n_new,
        /*.token     =*/ nullptr,
        /*.embd      =*/ const_cast<float *>(features),
        /*.pos       =*/ nullptr,
        /*.n_seq_id  =*/ nullptr,
        /*.seq_id    =*/ nullptr,
        /*.logits    =*/ nullptr,
    };
    GGML_ASSERT(llama_encode(ctx_encoder, enc_batch) == 0);

    const float * g_embd = llama_get_embeddings(ctx_encoder);
    GGML_ASSERT(g_embd != nullptr);

    llama_set_eagle3_g_embeddings(ctx_decoder, g_embd, n_embd, n_new);

    common_batch_clear(batch);
    for (int i = 0; i < n_new; ++i) {
        const int pos = spec->eagle3_n_past + i;
        const llama_token tok = (pos < n - 1) ? prompt_tgt[pos + 1] : id_last;
        common_batch_add(batch, tok, pos, { 0 }, true);
    }
    GGML_ASSERT(llama_decode(ctx_decoder, batch) == 0);

    spec->eagle3_n_past = n;

    llama_tokens result;
    result.reserve(params.n_draft);
    common_sampler_reset(smpl);

    auto sample_and_check = [&](int idx) -> bool {
        const llama_token id = common_sampler_sample(smpl, ctx_decoder, idx);
        const auto * cur_p = common_sampler_get_candidates(smpl, false);
        GGML_ASSERT(cur_p->selected != -1);
        const float p_sel = cur_p->data[cur_p->selected].p;

        common_sampler_accept(smpl, id, true);
        result.push_back(id);

        return std::isfinite(p_sel) && p_sel >= params.p_min;
    };

    if (!sample_and_check(n_new - 1)) {
        return result;
    }

    const float * prenorm = llama_get_embeddings_ith(ctx_decoder, -1);
    for (int i = 1; i < params.n_draft; ++i) {
        GGML_ASSERT(prenorm != nullptr);
        llama_set_eagle3_g_embeddings(ctx_decoder, prenorm, n_embd, 1);

        common_batch_clear(batch);
        common_batch_add(batch, result.back(), n - 1 + i, { 0 }, true);
        GGML_ASSERT(llama_decode(ctx_decoder, batch) == 0);

        prenorm = llama_get_embeddings_ith(ctx_decoder, -1);
        if (!sample_and_check(0)) {
            break;
        }
    }

    return result;
}

llama_tokens common_speculative_gen_eagle3_topk(
        struct common_speculative * spec,
        const llama_tokens & prompt_tgt,
        llama_token id_last,
        int32_t top_k) {
    auto * ctx_tgt     = spec->ctx_tgt;
    auto * ctx_encoder = spec->eagle3_encoder;
    auto * ctx_decoder = spec->eagle3_decoder;
    auto & batch       = spec->batch;

    GGML_ASSERT(ctx_encoder != nullptr && ctx_decoder != nullptr);
    GGML_ASSERT(top_k > 0);

    const int n_embd = llama_model_n_embd(llama_get_model(ctx_encoder));
    const int n = (int) prompt_tgt.size();
    const int n_new = n - spec->eagle3_n_past;

    GGML_ASSERT(n >= 1);
    GGML_ASSERT(n_new >= 1);

    llama_memory_seq_rm(llama_get_memory(ctx_decoder), 0, spec->eagle3_n_past, -1);

    const float * features = llama_get_eagle3_target_features(ctx_tgt);
    GGML_ASSERT(features != nullptr && "no EAGLE3 target features");
    const size_t feature_size = llama_get_eagle3_target_features_size(ctx_tgt);
    const size_t feature_row_size = (size_t) n_embd * 3;
    GGML_ASSERT(feature_size >= feature_row_size * (size_t) n_new &&
            "not enough EAGLE3 target feature rows for draft encoder");

    llama_batch enc_batch = {
        /*.n_tokens  =*/ n_new,
        /*.token     =*/ nullptr,
        /*.embd      =*/ const_cast<float *>(features),
        /*.pos       =*/ nullptr,
        /*.n_seq_id  =*/ nullptr,
        /*.seq_id    =*/ nullptr,
        /*.logits    =*/ nullptr,
    };
    GGML_ASSERT(llama_encode(ctx_encoder, enc_batch) == 0);

    const float * g_embd = llama_get_embeddings(ctx_encoder);
    GGML_ASSERT(g_embd != nullptr);
    llama_set_eagle3_g_embeddings(ctx_decoder, g_embd, n_embd, n_new);

    common_batch_clear(batch);
    for (int i = 0; i < n_new; ++i) {
        const int pos = spec->eagle3_n_past + i;
        const llama_token tok = (pos < n - 1) ? prompt_tgt[pos + 1] : id_last;
        common_batch_add(batch, tok, pos, { 0 }, true);
    }
    GGML_ASSERT(llama_decode(ctx_decoder, batch) == 0);
    spec->eagle3_n_past = n;

    const float * logits = llama_get_logits_ith(ctx_decoder, n_new - 1);
    GGML_ASSERT(logits != nullptr);

    const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx_tgt)));
    std::vector<llama_token> ids(n_vocab);
    std::iota(ids.begin(), ids.end(), 0);
    top_k = std::min(top_k, n_vocab);
    std::partial_sort(ids.begin(), ids.begin() + top_k, ids.end(),
            [&](llama_token a, llama_token b) { return logits[a] > logits[b]; });
    ids.resize(top_k);
    return ids;
}


static llama_tokens gen_dflash_draft(
        common_speculative * spec,
        common_speculative_params params,
        const llama_tokens & prompt_tgt,
        llama_token id_last) {
    llama_context * ctx_tgt = spec->ctx_tgt;
    llama_context * ctx_dft = spec->ctx_dft;
    llama_memory_t mem_dft = llama_get_memory(ctx_dft);

    const int32_t n = static_cast<int32_t>(prompt_tgt.size());
    const int32_t n_new = n - spec->dflash_n_past;
    GGML_ASSERT(n_new >= 0);
    llama_memory_seq_rm(mem_dft, 0, spec->dflash_n_past, -1);

    if (n_new > 0) {
        const float * features = llama_get_eagle3_target_features(ctx_tgt);
        GGML_ASSERT(features && "DFlash target features are not available");

        // This Orin branch's batched 10240 -> 2048 encoder projection can
        // silently return zero rows. Singleton calls use the correct matmul
        // path and generation normally commits only a few positions per step.
        constexpr int32_t n_chunk = 1;
        for (int32_t offset = 0; offset < n_new; offset += n_chunk) {
            llama_batch enc_batch = {
                /*.n_tokens  =*/ n_chunk,
                /*.token     =*/ nullptr,
                /*.embd      =*/ const_cast<float *>(
                    features + static_cast<size_t>(offset) * spec->dflash_n_embd_enc),
                /*.pos       =*/ nullptr,
                /*.n_seq_id  =*/ nullptr,
                /*.seq_id    =*/ nullptr,
                /*.logits    =*/ nullptr,
            };
            GGML_ASSERT(llama_encode(ctx_dft, enc_batch) == 0);

            const float * g_embd = llama_get_embeddings(ctx_dft);
            GGML_ASSERT(g_embd && "DFlash encoder produced no output");

            llama_batch & inject = spec->dflash_batch_inject;
            inject.n_tokens = n_chunk;
            std::memcpy(inject.embd, g_embd,
                static_cast<size_t>(n_chunk) * spec->dflash_n_embd_dec * sizeof(float));
            for (int32_t i = 0; i < n_chunk; ++i) {
                inject.pos[i] = spec->dflash_n_past + offset + i;
                inject.n_seq_id[i] = 1;
                inject.seq_id[i][0] = 0;
                // The shared DFlash context has embeddings enabled, so its
                // batch allocator requires every embedding row as an output.
                inject.logits[i] = true;
            }
            GGML_ASSERT(llama_decode(ctx_dft, inject) == 0);
        }
    }
    spec->dflash_n_past = n;

    const int32_t n_draft = std::max(0, std::min(params.n_draft, spec->dflash_block_size - 1));
    if (n_draft == 0) {
        return {};
    }

    llama_batch & batch = spec->batch;
    common_batch_clear(batch);
    common_batch_add(batch, id_last, n, {0}, true);
    for (int32_t i = 0; i < n_draft; ++i) {
        common_batch_add(batch, spec->dflash_mask_token, n + i + 1, {0}, true);
    }
    GGML_ASSERT(llama_decode(ctx_dft, batch) == 0);

    llama_tokens result;
    result.reserve(n_draft);
    common_sampler_reset(spec->smpl);
    for (int32_t i = 1; i <= n_draft; ++i) {
        common_sampler_sample(spec->smpl, ctx_dft, i);
        const llama_token_data_array * candidates = common_sampler_get_candidates(spec->smpl, true);
        GGML_ASSERT(candidates->size > 0);
        const llama_token id = candidates->data[0].id;
        const float probability = candidates->data[0].p;
        common_sampler_accept(spec->smpl, id, true);
        result.push_back(id);
        if (!std::isfinite(probability) || probability < params.p_min) {
            break;
        }
    }
    return result;
}

llama_tokens common_speculative_gen_draft(
        struct common_speculative * spec,
        struct common_speculative_params params,
        const llama_tokens & prompt_tgt_main_model, // specified in target model vocab
        llama_token id_last) {
    if (spec->eagle3_encoder != nullptr && spec->eagle3_decoder != nullptr) {
        return common_speculative_gen_eagle3_draft(spec, params, prompt_tgt_main_model, id_last);
    }
    if (spec->dflash) {
        return gen_dflash_draft(spec, params, prompt_tgt_main_model, id_last);
    }

    auto & batch  = spec->batch;
    auto & ctx_tgt = spec->ctx_tgt;
    auto & ctx_dft = spec->ctx_dft;
    auto & smpl   = spec->smpl;
    auto & prompt_dft = spec->prompt_dft;

    auto * mem_dft = llama_get_memory(ctx_dft);

    int reuse_i = 0;
    int reuse_n = 0;

    const int n_ctx = llama_n_ctx(ctx_dft) - params.n_draft;

    llama_tokens prompt_tgt_draft_model;
    if (!spec->vocab_dft_compatible) {
        std::string text;
        text = common_detokenize(ctx_tgt, prompt_tgt_main_model, true);
        text = replace_to_dft(spec, text);
        LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());
        prompt_tgt_draft_model = common_tokenize(ctx_dft, text, false, true);

        // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
        const auto * model_tgt = llama_get_model(ctx_tgt);
        const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

        int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
        GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");
        text.resize(-n_chars);
        llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
        text = replace_to_dft(spec, text);

        LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
        id_last = common_tokenize(ctx_dft, text, false, true)[0];
    }
    // prompt_tgt's tokens will always be compatible with ctx_dft
    const llama_tokens &prompt_tgt =
        spec->vocab_dft_compatible ? prompt_tgt_main_model : prompt_tgt_draft_model;

    const int i_start = std::max<int>(0, (int) prompt_tgt.size() - n_ctx);

    // reuse as much as possible from the old draft context
    // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
    for (int i = 0; i < (int) prompt_dft.size(); ++i) {
        int cur = 0;
        while (i_start + cur < (int) prompt_tgt.size() &&
               i       + cur < (int) prompt_dft.size() &&
               prompt_tgt[i_start + cur] == prompt_dft[i + cur]) {
            cur++;
        }

        if ((cur >= params.n_reuse || n_ctx >= (int) prompt_tgt.size()) && cur > reuse_n) {
            reuse_i = i;
            reuse_n = cur;
        }
    }

    LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

    llama_tokens result;
    result.reserve(params.n_draft);

    if (reuse_n == 0) {
        llama_memory_clear(mem_dft, false);
        prompt_dft.clear();
    } else {
        // this happens when a previous draft has been discarded (for example, due to being too small), but the
        // target model agreed with it. in this case, we simply pass back the previous results to save compute
        if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
            for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                result.push_back(prompt_dft[i]);

                if (params.n_draft <= (int) result.size()) {
                    break;
                }
            }

            return result;
        }

        if (reuse_i > 0) {
            llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
            llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

            prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
        }

        if (reuse_n < (int) prompt_dft.size()) {
            llama_memory_seq_rm (mem_dft, 0, reuse_n, -1);
            prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
        }
    }

    // prepare a batch to evaluate any new tokens in the prompt
    common_batch_clear(batch);

    for (size_t i = i_start + reuse_n; i < prompt_tgt.size(); ++i) {
        //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_tgt[i]);
        common_batch_add(batch, prompt_tgt[i], i - i_start, { 0 }, false);

        prompt_dft.push_back(prompt_tgt[i]);
    }

    // we should rarely end-up here during normal decoding
    if (batch.n_tokens > 0) {
        //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());

        llama_decode(ctx_dft, batch);
    }

    const llama_pos n_past = prompt_dft.size();

    LOG_DBG("%s: n_past = %d\n", __func__, n_past);

    common_batch_clear(batch);
    common_batch_add  (batch, id_last, n_past, { 0 }, true);

    prompt_dft.push_back(id_last);

    LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

    llama_decode(ctx_dft, batch);

    common_sampler_reset(smpl);

    // sample n_draft tokens from the draft model
    for (int i = 0; i < params.n_draft; ++i) {
        common_batch_clear(batch);

        common_sampler_sample(smpl, ctx_dft, 0, true);

        const auto * cur_p = common_sampler_get_candidates(smpl, true);

        for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
            LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                    k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
        }

        // add drafted token for each sequence
        const llama_token id = cur_p->data[0].id;

        common_sampler_accept(smpl, id, true);

        result.push_back(id);

        if (params.n_draft <= (int) result.size()) {
            break;
        }

        // only collect very high-confidence draft tokens
        if (cur_p->data[0].p < params.p_min) {
            break;
        }

        common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

        // evaluate the drafted tokens on the draft model
        llama_decode(ctx_dft, batch);

        prompt_dft.push_back(id);
    }

    if (!spec->vocab_dft_compatible) {
        std::string detokenized = common_detokenize(ctx_dft, result, true);
        detokenized = replace_to_tgt(spec, detokenized);
        LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
        result = common_tokenize(ctx_tgt, detokenized, false, true);
        if (result.size() > (size_t)params.n_draft) {
            result.resize(params.n_draft);
        }
    }
    return result;
}
