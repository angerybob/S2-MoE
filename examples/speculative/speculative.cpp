#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <random>
#include <regex>
#include <set>
#include <string>
#include <vector>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

struct seq_draft {
    bool active   = false;
    bool drafting = false;
    bool skip     = false;

    int i_batch_dft = 0;
    std::vector<int> i_batch_tgt;

    std::vector<llama_token> tokens;
    // ================= [新增] =================
    std::vector<float> probs;  // 存储每个 token 的原始概率 P(x|ctx)
    // ==========================================
    std::vector<std::vector<llama_token_data>> dists;

    struct common_sampler * smpl = nullptr;
};

struct spec_profile {
    int64_t draft_sample_us = 0;
    int64_t target_sample_us = 0;
    int64_t draft_decode_us = 0;
    int64_t target_decode_us = 0;
    int64_t memory_us = 0;

    int64_t draft_sample_calls = 0;
    int64_t target_sample_calls = 0;
    int64_t draft_decode_calls = 0;
    int64_t target_decode_calls = 0;
    int64_t draft_decode_tokens = 0;
    int64_t target_decode_tokens = 0;
};

struct prune_profile {
    int64_t steps = 0;
    int64_t stopped = 0;
    int64_t empty_topk = 0;
    int64_t total_token_experts = 0;
    int64_t total_resident_experts = 0;
    int64_t total_offchip_experts = 0;
    int64_t total_new_offchip_experts = 0;
    double  total_delta_g = 0.0;
    double  total_delta_c = 0.0;
    double  total_decode_ms = 0.0;
};

struct expert_pair_cmp {
    bool operator()(const std::pair<int, int> & a, const std::pair<int, int> & b) const {
        return a.first < b.first || (a.first == b.first && a.second < b.second);
    }
};

static std::set<std::pair<int, int>, expert_pair_cmp> capture_moe_experts(llama_context * ctx, const llama_model * model, int idx_in_batch) {
    std::set<std::pair<int, int>, expert_pair_cmp> result;

    const int n_layers = llama_model_n_layer(model);
    std::vector<llama_moe_topk_layer> topk_layers(n_layers);
    int n_out_layers = 0;
    if (llama_get_last_moe_topk(ctx, idx_in_batch, topk_layers.data(), n_layers, &n_out_layers)) {
        for (int i = 0; i < n_out_layers; ++i) {
            for (int j = 0; j < topk_layers[i].n; ++j) {
                if (topk_layers[i].expert_id[j] >= 0) {
                    result.insert({topk_layers[i].layer, topk_layers[i].expert_id[j]});
                }
            }
        }
    }
    if (!result.empty()) {
        return result;
    }

    const int32_t n_total = llama_moe_expert_capture_get(nullptr, nullptr, 0);
    if (n_total <= 0) {
        return result;
    }

    std::vector<int32_t> layers(n_total);
    std::vector<int32_t> experts(n_total);
    const int32_t n_read = llama_moe_expert_capture_get(layers.data(), experts.data(), n_total);
    for (int32_t i = 0; i < std::min(n_total, n_read); ++i) {
        if (layers[i] >= 0 && experts[i] >= 0) {
            result.insert({layers[i], experts[i]});
        }
    }

    return result;
}

static float token_softmax_prob_from_logits(llama_context * ctx, int idx_in_batch, llama_token token_id) {
    const float * logits = llama_get_logits_ith(ctx, idx_in_batch);
    if (logits == nullptr || token_id < 0) {
        return 1.0f;
    }

    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (token_id >= n_vocab) {
        return 1.0f;
    }

    float max_logit = -INFINITY;
    for (int i = 0; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum += std::exp((double) logits[i] - (double) max_logit);
    }

    if (sum <= 0.0 || !std::isfinite(sum)) {
        return 1.0f;
    }

    const double p = std::exp((double) logits[token_id] - (double) max_logit) / sum;
    if (!std::isfinite(p)) {
        return 1.0f;
    }

    return (float) std::max(0.0, std::min(1.0, p));
}

static float cumulative_prob(const std::vector<float> & probs, size_t start) {
    double p = 1.0;
    for (size_t i = start; i < probs.size(); ++i) {
        const float v = probs[i];
        p *= std::max(0.0f, std::min(1.0f, v));
        if (p <= 0.0) {
            return 0.0f;
        }
    }
    return (float) std::max(0.0, std::min(1.0, p));
}

static int count_resident_experts(
        const std::set<std::pair<int, int>, expert_pair_cmp> & token_experts,
        const std::set<std::pair<int, int>, expert_pair_cmp> & resident_experts) {
    int n = 0;
    for (const auto & expert : token_experts) {
        if (resident_experts.find(expert) != resident_experts.end()) {
            ++n;
        }
    }
    return n;
}

static int count_offchip_experts(
        const std::set<std::pair<int, int>, expert_pair_cmp> & token_experts,
        const std::set<std::pair<int, int>, expert_pair_cmp> & resident_experts) {
    return (int) token_experts.size() - count_resident_experts(token_experts, resident_experts);
}

static int count_new_experts(
        const std::set<std::pair<int, int>, expert_pair_cmp> & token_experts,
        const std::set<std::pair<int, int>, expert_pair_cmp> & seen_experts,
        const std::set<std::pair<int, int>, expert_pair_cmp> & resident_experts) {
    int n = 0;
    for (const auto & expert : token_experts) {
        if (seen_experts.find(expert) == seen_experts.end() &&
                resident_experts.find(expert) == resident_experts.end()) {
            ++n;
        }
    }
    return n;
}

static std::string format_expert_sample(
        const std::set<std::pair<int, int>, expert_pair_cmp> & token_experts,
        int max_items) {
    std::ostringstream ss;
    int n = 0;
    for (const auto & expert : token_experts) {
        if (n++ > 0) {
            ss << ",";
        }
        ss << "(" << expert.first << "," << expert.second << ")";
        if (n >= max_items) {
            if ((int) token_experts.size() > n) {
                ss << ",...";
            }
            break;
        }
    }
    return ss.str();
}

static std::set<std::pair<int, int>, expert_pair_cmp> load_resident_experts_json(const std::string & path) {
    std::set<std::pair<int, int>, expert_pair_cmp> result;
    if (path.empty()) {
        return result;
    }

    std::ifstream in(path);
    if (!in) {
        LOG_WRN("%s: failed to open %s; pruning will treat all experts as off-chip\n", __func__, path.c_str());
        return result;
    }

    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::regex pair_re("\\[\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*\\]");
    for (std::sregex_iterator it(text.begin(), text.end(), pair_re), end; it != end; ++it) {
        result.insert({std::stoi((*it)[1].str()), std::stoi((*it)[2].str())});
    }

    LOG_INF("%s: loaded %zu resident expert ids for pruning cost model from %s\n", __func__, result.size(), path.c_str());
    return result;
}

int main(int argc, char ** argv) {
    common_params params;

    // needed to get candidate probs even for temp <= 0.0
    params.sampling.n_probs = 128;

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

    // max number of parallel drafting sequences (i.e. tree branches)
    const int n_seq_dft = params.n_parallel;

    // probability threshold for splitting a draft branch (only for n_seq_dft > 1)
    const float p_draft_split = params.speculative.p_split;

    std::default_random_engine rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? std::random_device()() : params.sampling.seed);
    std::uniform_real_distribution<> u_dist;

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;
    llama_model * model_dft = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;
    llama_context_ptr ctx_dft_shared_model;

    // load the target model
    const auto params_tgt = params;
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    // load or attach the draft model
    common_params params_dft = params_tgt;
    params_dft.devices = params.speculative.devices;
    params_dft.model = params.speculative.model;
    params_dft.n_gpu_layers = params.speculative.n_gpu_layers;
    params_dft.n_expert_used = params.speculative.n_expert_used;
    if (params.speculative.cpuparams.n_threads > 0) {
        params_dft.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
    }

    params_dft.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
    params_dft.tensor_buft_overrides     = params.speculative.tensor_buft_overrides;

    common_init_result llama_init_dft;
    const bool same_model_weights = params_dft.model.path == params_tgt.model.path &&
                                    params_dft.lora_adapters.empty() &&
                                    params_dft.control_vectors.empty();

    if (same_model_weights) {
        auto cparams_dft = common_context_params_to_llama(params_dft);
        ctx_dft_shared_model.reset(llama_init_from_model(model_tgt, cparams_dft));
        if (!ctx_dft_shared_model) {
            LOG_ERR("%s: failed to create draft context from shared target model '%s'\n", __func__, params_tgt.model.path.c_str());
            return 1;
        }
        model_dft = model_tgt;
        ctx_dft   = ctx_dft_shared_model.get();
        if (params_dft.warmup) {
            const llama_vocab * vocab = llama_model_get_vocab(model_dft);
            std::vector<llama_token> tmp;
            llama_token bos = llama_vocab_bos(vocab);
            llama_token eos = llama_vocab_eos(vocab);
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
        model_dft = llama_init_dft.model.get();
        ctx_dft   = llama_init_dft.context.get();
    }

    if (model_dft == nullptr || ctx_dft == nullptr) {
        LOG_ERR("%s: failed to initialize draft model/context\n", __func__);
        return 1;
    }
    llama_context_set_draft_context(ctx_dft, true);
    if (params.speculative.prune != 0) {
        llama_set_moe_topk(ctx_dft, true);
        LOG_INF("%s: marginal MoE draft pruning enabled: max_depth=%d max_nodes=%d budget=%.3f tpot=%.3f expert=%.3f MiB bw=%.3f MiB/ms eps=%.3f score=%.3f conf=%.3f\n",
                __func__,
                params.speculative.prune_max_depth,
                params.speculative.prune_max_nodes,
                (double) params.sprune.budget_b,
                (double) params.sprune.tpot,
                (double) params.sprune.expert_bytes,
                (double) params.sprune.bandwidth,
                (double) params.sprune.eps,
                (double) params.sprune.score_thresh,
                (double) params.sprune.conf_thresh);
    }
    const auto prune_resident_experts =
        params.speculative.prune != 0 ? load_resident_experts_json(params.gpu_experts_json)
                                      : std::set<std::pair<int, int>, expert_pair_cmp>();
    // =============== 在这里插入修改 ===============

    // 检查是否适合共享 KV Cache (Self-Speculation)
    // 条件：Draft 和 Target 架构一致 (层数、维度等)
    if (llama_n_layer(model_tgt) == llama_n_layer(model_dft) &&
        llama_n_embd(model_tgt) == llama_n_embd(model_dft) &&
        llama_n_head(model_tgt) == llama_n_head(model_dft) && params.speculative.share_kv) {
        
        LOG_INF("%s: Enabling Self-Speculation Optimization (Shared KV Cache)\n", __func__);
        llama_share_kv_cache(ctx_dft, ctx_tgt);
        
    } else {
        LOG_INF("%s: Models are different, using independent KV Caches.\n", __func__);
    }

    // ===========================================
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("vocab_type tgt: %d\n", vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("vocab_type dft: %d\n", vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_ERR("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_ERR("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return 1;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_ERR("%s: draft model special tokens must match target model to use speculation\n", __func__);
        return 1;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_ERR("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_ERR("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return 1;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);
            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_ERR("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_ERR("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return 1;
            }
        }
    }

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    const int max_context_size     = llama_n_ctx(ctx_tgt);
    const int max_tokens_list_size = max_context_size - 4;

    if ((int) inp.size() > max_tokens_list_size) {
        LOG_ERR("%s: prompt too long (%d tokens, max %d)\n", __func__, (int) inp.size(), max_tokens_list_size);
        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    const int n_input = inp.size();

    const auto t_enc_start = ggml_time_us();

    // eval the prompt with both models
    // 1. Target 处理 Prompt (这会填充共享的 KV Cache)
    llama_decode(ctx_tgt, llama_batch_get_one( inp.data(), n_input - 1));
    llama_decode(ctx_tgt, llama_batch_get_one(&inp.back(),           1));

    // 2. Draft 处理 Prompt
    // === [修改] 注释掉下面这一行 ===
    // 在共享 KV 模式下，Target 已经填好了显存，Draft 直接用即可。
    // 如果不注释，会触发 discontinuous sequence 错误。
    // llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input));
    // 建议加上判断，为了代码兼容性：
    // 如果没有共享 (指针不同)，才执行 decode
    if (llama_get_memory(ctx_tgt) != llama_get_memory(ctx_dft)) {
         llama_decode(ctx_dft, llama_batch_get_one( inp.data(), n_input - 1));
         llama_decode(ctx_dft, llama_batch_get_one(&inp.back(),           1));
    }
    const auto t_enc_end = ggml_time_us();

    // the 2 models should have the same vocab
    //GGML_ASSERT(n_vocab == llama_vocab_n_tokens(model_dft));

    // how many tokens to draft each time
    const int n_draft_max = params.speculative.n_max;
    const int n_draft_min = std::max(1, params.speculative.n_min);
    int n_draft = n_draft_max;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;
    spec_profile prof;
    prune_profile pprof;

    int n_past_tgt = inp.size();
    int n_past_dft = inp.size();

    // used to determine end of generation
    bool has_eos = false;

    // target model sampling context (reuse the llama_context's sampling instance)
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // draft sequence data
    std::vector<seq_draft> drafts(n_seq_dft);

    for (int s = 0; s < n_seq_dft; ++s) {
        // allocate llama_sampler for each draft sequence
        drafts[s].smpl = common_sampler_init(model_dft, params.sampling);
    }

    llama_batch batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, n_seq_dft);

    const auto t_dec_start = ggml_time_us();

    // sample from the last token of the prompt
    drafts[0].i_batch_tgt.resize(1);
    drafts[0].i_batch_tgt[0] = 0;

    while (true) {
        std::set<int> active_seqs = {};

        // print current draft sequences
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            active_seqs.insert(s);
            const auto & tokens = drafts[s].tokens;

            LOG_DBG("draft %d: %s\n", s, string_from(ctx_dft, tokens).c_str());
        }

        int i_dft  = 0;
        int s_keep = 0;

        llama_token token_id;
        std::string token_str;

        // loop until we fail to accept a drafted token or we run out of drafted tokens
        while (true) {

            // check if the target token matches any of the drafts
            // for stochastic sampling, attempt to match the token with the drafted tokens
            {
                bool accept = false;
                if (params.sampling.temp > 0) {
                    // stochastic verification
                    const int64_t t_sample_tgt_start = ggml_time_us();
                    common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft], true);
                    prof.target_sample_us += ggml_time_us() - t_sample_tgt_start;
                    prof.target_sample_calls++;

                    auto & dist_tgt = *common_sampler_get_candidates(smpl, true);

                    float p_tgt = 0.0f;
                    float p_dft = 0.0f;

                    while (active_seqs.size() > 0) {
                        // randomly select a sequence to verify from active sequences
                        std::uniform_int_distribution<unsigned int> u_int_dist(0, active_seqs.size() - 1);
                        int s = *std::next(active_seqs.begin(), u_int_dist(rng));
                        if (i_dft >= (int) drafts[s].tokens.size()) {
                            drafts[s].active = false;
                            active_seqs.erase(s);
                            continue;
                        }
                        if (accept) {
                            // if we already accepted a token, we can skip the rest
                            if (drafts[s].tokens[i_dft] != drafts[s_keep].tokens[i_dft]) {
                                drafts[s].active = false;
                                active_seqs.erase(s);
                            }
                            continue;
                        }

                        LOG_DBG("verifying sequence #%d at pos #%d from %d active sequence(s)\n", s, i_dft, (int) active_seqs.size());
                        float r = u_dist(rng);
                        llama_token_data_array dist_dft = { drafts[s].dists[i_dft].data() , drafts[s].dists[i_dft].size(), LLAMA_TOKEN_NULL, true };

                        //GGML_ASSERT(dist_tgt.size <= dist_dft.size);

                        // acquire the token probabilities assigned by the draft and target models
                        for (size_t i = 0; i < dist_tgt.size; i++) {
                            if (dist_tgt.data[i].id == drafts[s].tokens[i_dft]) {
                                p_tgt = dist_tgt.data[i].p;
                                break;
                            }
                        }
                        for (size_t i = 0; i < dist_dft.size; i++) {
                            if (dist_dft.data[i].id == drafts[s].tokens[i_dft]) {
                                p_dft = dist_dft.data[i].p;
                                break;
                            }
                        }
                        // [计算累积概率]
                        // 也就是 Draft 生成这个 token 时的 confidence * 之前所有 token 的 confidence
                        float cum_prob = 1.0f;
                        // 注意：i_dft 是当前验证到的位置。
                        // drafts[s].probs 存的是 Draft 对每个 token 的预测概率。
                        // 我们只乘到 i_dft 为止。
                        for (int k = 0; k <= i_dft && k < (int)drafts[s].probs.size(); ++k) {
                            cum_prob *= drafts[s].probs[k];
                        }
                        
                        // 获取当前单个 Token 的概率
                        float cur_prob = (i_dft < (int)drafts[s].probs.size()) ? drafts[s].probs[i_dft] : 0.0f;
                        LOG_DBG("r = %f, p_dft = %f, p_tgt = %f\n", r, p_dft, p_tgt);
                        if (r <= p_tgt / p_dft) {
                            s_keep = s;
                            accept = true;
                            token_id = drafts[s].tokens[i_dft];
                            token_str = common_token_to_piece(ctx_tgt, token_id);
                            common_sampler_accept(smpl, token_id, true);
                            // ================= [日志输出：接收] =================
                            // 使用 Cyan 色 (\033[36m) 
                            if (params.speculative.accept_log) {
                                fprintf(stderr, "\033[36m[ANALYSIS] Token: '%s' | Prob: %.4f | CumProb: %.4f | Status: ACCEPT\033[0m\n",
                                        token_str.c_str(), cur_prob, cum_prob);
                            }

                            // ====================================================
                            LOG_DBG("draft token %d of sequence %d (%d, '%s') accepted\n", i_dft, s, token_id, token_str.c_str());
                            break;
                        } else {
                            LOG_DBG("draft token %d of sequence %d (%d, '%s') rejected\n", i_dft, s, drafts[s].tokens[i_dft], common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str());
                            drafts[s].active = false;
                            // ================= [日志输出：拒绝] =================
                            // 使用 Red 色 (\033[31m)
                            if (params.speculative.accept_log) {
                                fprintf(stderr, "\033[31m[ANALYSIS] Token: '%s' | Prob: %.4f | CumProb: %.4f | Status: REJECT\033[0m\n",
                                        common_token_to_piece(ctx_tgt, drafts[s].tokens[i_dft]).c_str(), 
                                        cur_prob, cum_prob);
                            }
                            // ====================================================
                            // calculate residual probability
                            GGML_ASSERT(dist_tgt.sorted);
                            GGML_ASSERT(dist_dft.sorted);

                            // sort dist by id
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });
                            std::sort(dist_dft.data, dist_dft.data + dist_dft.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.id < b.id;
                            });

                            float sum_probs = 0.0f;

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                if (i < dist_dft.size) {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p - dist_dft.data[i].p);
                                } else {
                                    dist_tgt.data[i].p = std::max(0.0f, dist_tgt.data[i].p);
                                }

                                sum_probs += dist_tgt.data[i].p;
                            }

                            for (size_t i = 0; i < dist_tgt.size; i++) {
                                dist_tgt.data[i].p /= sum_probs;
                            }

                            // sort dist_tgt by p desc
                            std::sort(dist_tgt.data, dist_tgt.data + dist_tgt.size, [](const llama_token_data &a, const llama_token_data &b) {
                                return a.p > b.p;
                            });
                        }

                        active_seqs.erase(s);
                        for (int i = 0; i < n_seq_dft; i++) {
                            if (i == s) {
                                continue;
                            }
                            if (drafts[i].active && drafts[i].tokens[i_dft] == drafts[s].tokens[i_dft]) {
                                // synchronize active status for sequences with the same drafted token
                                drafts[i].active = drafts[i].active && accept;
                                if (!drafts[i].active) {
                                    active_seqs.erase(s);
                                }
                            }
                        }
                    }

                    if (!accept) {
                        // all drafted tokens were rejected
                        // sample from the target model
                        LOG_DBG("all drafted tokens were rejected, sampling from residual distribution\n");
                        std::vector<float> probs(dist_tgt.size);
                        for (size_t i = 0; i < dist_tgt.size; ++i) {
                            probs[i] = dist_tgt.data[i].p;
                        }

                        std::discrete_distribution<> dist(probs.begin(), probs.end());

                        const int idx = dist(rng);

                        token_id = dist_tgt.data[idx].id;
                        common_sampler_accept(smpl, token_id, true);
                        token_str = common_token_to_piece(ctx_tgt, token_id);
                    }
                } else {
                    // greedy verification

                    // sample from the target model
                    LOG_DBG("sampling target: s_keep = %3d, i_dft = %3d, i_batch_tgt = %3d\n", s_keep, i_dft, drafts[s_keep].i_batch_tgt[i_dft]);
                    const int64_t t_sample_tgt_start = ggml_time_us();
                    token_id = common_sampler_sample(smpl, ctx_tgt, drafts[s_keep].i_batch_tgt[i_dft]);
                    prof.target_sample_us += ggml_time_us() - t_sample_tgt_start;
                    prof.target_sample_calls++;

                    common_sampler_accept(smpl, token_id, true);

                    token_str = common_token_to_piece(ctx_tgt, token_id);

                    for (int s = 0; s < n_seq_dft; ++s) {
                        if (!drafts[s].active) {
                            continue;
                        }

                        if (i_dft < (int) drafts[s].tokens.size() && token_id == drafts[s].tokens[i_dft]) {
                            LOG_DBG("the sampled target token matches the %dth drafted token of sequence %d (%d, '%s') - accepted\n", i_dft, s, token_id, token_str.c_str());

                            s_keep = s;
                            accept = true;
                        } else {
                            drafts[s].active = false;
                        }
                    }
                }

                if (llama_vocab_is_eog(vocab_tgt, token_id)) {
                    has_eos = true;
                }
                ++n_predict;

                if (accept) {
                    ++n_accept;
                    ++n_past_tgt;
                    ++n_past_dft;
                    ++i_dft;
                    if (params.use_color) {
                        // Color token according to its origin sequence
                        LOG("\u001b[%dm%s\u001b[37m", (36 - s_keep % 6), token_str.c_str());
                    } else {
                        LOG("%s", token_str.c_str());
                    }
                    continue;
                } else {
                    LOG("%s", token_str.c_str());
                    break;
                }
            }
        }
        // ======================= [新增日志代码开始] =======================
        // i_dft 就是这一轮被接受的 token 数量
        // drafts[0].tokens.size() 大致是这一轮草稿生成的总长度
        // 使用 stderr 输出，配合颜色代码，方便在控制台一眼看到，不干扰正文生成
        if (n_predict > 0 && params.speculative.accept_log) { // 避免在刚开始处理 prompt 时打印
            int total_drafted = drafts[0].tokens.size(); // 草稿总长
            fprintf(stderr, "\n\033[1;33m[Speculative Stats] Drafted: %d | Accepted: %d | Rate: %.1f%%\033[0m", 
                    total_drafted, 
                    i_dft, 
                    total_drafted > 0 ? (100.0f * i_dft / total_drafted) : 0.0f);
        }
        // ======================= [新增日志代码结束] =======================
        {
            LOG_DBG("the sampled target token (%d, '%s') did not match, or we ran out of drafted tokens\n", token_id, token_str.c_str());

            // TODO: simplify
            {
                LOG_DBG("keeping sequence %d, n_past_tgt = %d, n_past_dft = %d\n", s_keep, n_past_tgt, n_past_dft);

                const int64_t t_memory_start = ggml_time_us();
                llama_memory_seq_keep(mem_dft, s_keep);
                llama_memory_seq_cp  (mem_dft, s_keep, 0, -1, -1);
                llama_memory_seq_keep(mem_dft, 0);
                // === [优化] 如果内存共享，不需要删两次 ===
                if (mem_tgt != mem_dft) { 
                    llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);
                    llama_memory_seq_keep(mem_tgt, s_keep);
                    llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
                    llama_memory_seq_keep(mem_tgt, 0);
                }
                prof.memory_us += ggml_time_us() - t_memory_start;
                // llama_memory_seq_rm  (mem_tgt, s_keep, n_past_tgt, -1);
                // llama_memory_seq_keep(mem_tgt, s_keep);
                // llama_memory_seq_cp  (mem_tgt, s_keep, 0, -1, -1);
                // llama_memory_seq_keep(mem_tgt, 0);
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].active = false;
                drafts[s].tokens.clear();
                drafts[s].i_batch_tgt.clear();
                drafts[s].dists.clear();
            }
            // note: will be erased after the speculation phase
            drafts[0].tokens.push_back(token_id);
            // ================= [新增] =================
            // 这个 token 是 Target 生成的，对于 Draft 来说是已知条件，概率设为 1.0
            // 或者如果不参与计算累积概率，这一步其实不重要，但为了对齐 vector 长度必须加
            drafts[0].probs.push_back(1.0f); 
            // ==========================================
            drafts[0].dists.push_back(std::vector<llama_token_data>());
            drafts[0].i_batch_tgt.push_back(0);

            common_batch_clear(batch_dft);
            common_batch_add  (batch_dft, token_id, n_past_dft, { 0 }, true);

            const int64_t t_memory_start = ggml_time_us();
            if (n_drafted > 0) {
                llama_memory_seq_rm(mem_dft, 0, n_past_dft, -1);
            }
            prof.memory_us += ggml_time_us() - t_memory_start;
            // LOG_DBG("dft batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_dft, batch_dft).c_str());
            const int64_t t_decode_dft_start = ggml_time_us();
            const int ret_dft_feed = llama_decode(ctx_dft, batch_dft);
            if (ret_dft_feed != 0) {
                LOG_ERR("%s: draft feed decode failed: ret=%d token=%d pos=%d\n", __func__, ret_dft_feed, token_id, n_past_dft);
                return 1;
            }
            if (getenv("LLAMA_SPEC_DEBUG_DRAFT_FEED") != nullptr) {
                const float * logits = llama_get_logits_ith(ctx_dft, 0);
                const int n_vocab = llama_vocab_n_tokens(vocab_dft);
                int best = 0;
                for (int i = 1; i < n_vocab; ++i) {
                    if (logits[i] > logits[best]) {
                        best = i;
                    }
                }
                LOG_INF("debug draft feed: fed=%d '%s' pos=%d next_top=%d '%s' logit=%.6f\n",
                        token_id, common_token_to_piece(ctx_dft, token_id).c_str(), n_past_dft,
                        best, common_token_to_piece(ctx_dft, best).c_str(), (double) logits[best]);
            }
            prof.draft_decode_us += ggml_time_us() - t_decode_dft_start;
            prof.draft_decode_calls++;
            prof.draft_decode_tokens += batch_dft.n_tokens;

            ++n_past_dft;
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }

        if (i_dft >= n_draft) {
            n_draft = std::min(n_draft_max, n_draft + 2);
        } else if (i_dft <= n_draft / 2) {
            n_draft = std::max(n_draft_min, i_dft + 1);
        } else {
            n_draft = std::max(n_draft_min, std::min(n_draft_max, i_dft + 2));
        }

        if (drafts[0].smpl) {
            common_sampler_free(drafts[0].smpl);
        }
        drafts[0].smpl = common_sampler_clone(smpl);

        int n_seq_cur  = 1;
        int n_past_cur = n_past_dft;

        for (int s = 0; s < n_seq_dft; ++s) {
            drafts[s].active   = false;
            drafts[s].drafting = false;
        }
        drafts[0].active      = true;
        drafts[0].drafting    = true;
        drafts[0].i_batch_dft = 0;

        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, drafts[0].tokens[0], n_past_tgt, { 0 }, true);
        std::set<std::pair<int, int>, expert_pair_cmp> prune_seen_experts;
        float prune_budget_used = 0.0f;
        int prune_nodes = 0;
        const size_t prune_prob_start = drafts[0].probs.size();

        llama_set_ssd_cuda_cache_mode(1);

        // sample n_draft tokens from the draft model using tree-based sampling
        for (int i = 0; i < n_draft; ++i) {
            batch_dft.n_tokens = 0;
            bool stop_drafting = false;

            for (int s = 0; s < n_seq_dft; ++s) {
                drafts[s].skip = false;
            }

            for (int s = 0; s < n_seq_dft; ++s) {
                if (!drafts[s].drafting || drafts[s].skip) {
                    continue;
                }

                const int64_t t_sample_dft_start = ggml_time_us();
                common_sampler_sample(drafts[s].smpl, ctx_dft, drafts[s].i_batch_dft, true);
                prof.draft_sample_us += ggml_time_us() - t_sample_dft_start;
                prof.draft_sample_calls++;

                const auto * cur_p = common_sampler_get_candidates(drafts[s].smpl, true);
                if (i >= params.speculative.n_min && cur_p->size > 0 && cur_p->data[0].p < params.speculative.p_min) {
                    drafts[s].drafting = false;
                    stop_drafting = true;
                    continue;
                }

                for (int k = 0; k < std::min(n_seq_dft + 3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - draft candidate %3d for seq %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            k, s, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                std::vector<int> sa(1, s);

                // attempt to split the branch if the probability is high enough
                for (int f = 1; f < 8; ++f) {
                    if (n_seq_cur < n_seq_dft && cur_p->data[f].p > p_draft_split) {
                        LOG_DBG("splitting seq %3d into %3d\n", s, n_seq_cur);

                        llama_memory_seq_rm(mem_dft,    n_seq_cur, -1, -1);
                        llama_memory_seq_cp(mem_dft, s, n_seq_cur, -1, -1);

                        // all previous tokens from this branch are now also part of the new branch
                        for (int t = 0; t < batch_tgt.n_tokens; ++t) {
                            for (int p = 0; p < batch_tgt.n_seq_id[t]; ++p) {
                                if (batch_tgt.seq_id[t][p] == s) {
                                    batch_tgt.seq_id[t][batch_tgt.n_seq_id[t]] = n_seq_cur;
                                    batch_tgt.n_seq_id[t]++;
                                    break;
                                }
                            }
                        }

                        // copy the draft state
                        drafts[n_seq_cur].active   = true;
                        drafts[n_seq_cur].drafting = true;
                        drafts[n_seq_cur].skip     = true;

                        drafts[n_seq_cur].tokens      = drafts[s].tokens;
                        // ================= [新增：复制概率历史] =================
                        drafts[n_seq_cur].probs       = drafts[s].probs;
                        // ========================================================
                        drafts[n_seq_cur].dists       = drafts[s].dists;
                        drafts[n_seq_cur].i_batch_dft = drafts[s].i_batch_dft;
                        drafts[n_seq_cur].i_batch_tgt = drafts[s].i_batch_tgt;

                        if (drafts[n_seq_cur].smpl) {
                            common_sampler_free(drafts[n_seq_cur].smpl);
                        }
                        drafts[n_seq_cur].smpl = common_sampler_clone(drafts[s].smpl);

                        sa.push_back(n_seq_cur);

                        n_seq_cur++;
                    } else {
                        break;
                    }
                }

                // add drafted token for each sequence
                for (int is = 0; is < (int) sa.size(); ++is) {
                    const int s = sa[is];
                    const llama_token id = cur_p->data[is].id;
                    const float p_token  = token_softmax_prob_from_logits(ctx_dft, drafts[s].i_batch_dft, id);

                    common_sampler_accept(drafts[s].smpl, id, true);

                    drafts[s].tokens.push_back(id);
                    // ================= [新增：存入概率] =================
                    drafts[s].probs.push_back(p_token);
                    // ===================================================

                    // save cur_p.data into drafts[s].dists
                    drafts[s].dists.push_back({cur_p->data, cur_p->data + cur_p->size});

                    // add unique drafted tokens to the target batch
                    drafts[s].i_batch_tgt.push_back(batch_tgt.n_tokens);

                    common_batch_add(batch_tgt, id, n_past_tgt + i + 1, { s }, true);

                    // add the token to the batch for batched decoding with the draft model
                    drafts[s].i_batch_dft = batch_dft.n_tokens;

                    common_batch_add(batch_dft, id, n_past_cur, { s }, true);

                    if (batch_tgt.n_tokens > n_draft) {
                        drafts[s].drafting = false;
                    }
                }
            }

            // no sequence is drafting anymore
            if (batch_dft.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            if (params.speculative.prune != 0 && n_seq_dft == 1) {
                llama_moe_expert_capture_clear();
            }
            const int64_t t_decode_dft_start = ggml_time_us();
            llama_decode(ctx_dft, batch_dft);
            const int64_t t_decode_dft_us = ggml_time_us() - t_decode_dft_start;
            prof.draft_decode_us += t_decode_dft_us;
            prof.draft_decode_calls++;
            prof.draft_decode_tokens += batch_dft.n_tokens;
            ++n_past_cur;
            ++n_drafted;

            if (params.speculative.prune != 0 && n_seq_dft == 1) {
                const int idx = drafts[0].i_batch_dft;
                const float conf = drafts[0].probs.size() <= prune_prob_start ? 1.0f : cumulative_prob(drafts[0].probs, prune_prob_start);
                const auto token_experts = capture_moe_experts(ctx_dft, model_dft, idx);
                const int n_resident_experts = count_resident_experts(token_experts, prune_resident_experts);
                const int n_offchip_experts = count_offchip_experts(token_experts, prune_resident_experts);
                const int n_new_experts = count_new_experts(token_experts, prune_seen_experts, prune_resident_experts);
                const float delta_g = conf * params.sprune.tpot;
                const float delta_c = n_new_experts * (params.sprune.expert_bytes / params.sprune.bandwidth);
                const float score = delta_g / (delta_c + params.sprune.eps);
                const bool reject_conf = i == 0 && params.sprune.conf_thresh > 0.0f && conf < params.sprune.conf_thresh;
                const bool reject_score = i > 0 && score < params.sprune.score_thresh;
                const bool reject_budget = prune_budget_used + delta_c > params.sprune.budget_b;
                const bool reject_depth = i + 1 > params.speculative.prune_max_depth;
                const bool reject_nodes = prune_nodes + 1 > params.speculative.prune_max_nodes;

                pprof.steps++;
                pprof.empty_topk += token_experts.empty() ? 1 : 0;
                pprof.total_token_experts += (int64_t) token_experts.size();
                pprof.total_resident_experts += n_resident_experts;
                pprof.total_offchip_experts += n_offchip_experts;
                pprof.total_new_offchip_experts += n_new_experts;
                pprof.total_delta_g += delta_g;
                pprof.total_delta_c += delta_c;
                pprof.total_decode_ms += t_decode_dft_us / 1000.0;

                if (params.speculative.prune >= 2) {
                    LOG_INF("prune trace: step=%d conf=%.4f experts=%zu resident=%d offchip=%d new_offchip=%d "
                            "delta_g=%.4f delta_c=%.4f score=%.4f budget=%.4f decode_ms=%.3f reject=%c%c%c%c%c\n",
                            i, (double) conf, token_experts.size(), n_resident_experts, n_offchip_experts, n_new_experts,
                            (double) delta_g, (double) delta_c, (double) score, (double) prune_budget_used,
                            t_decode_dft_us / 1000.0,
                            reject_conf ? 'C' : '-',
                            reject_score ? 'S' : '-',
                            reject_budget ? 'B' : '-',
                            reject_depth ? 'D' : '-',
                            reject_nodes ? 'N' : '-');
                    if (params.speculative.prune >= 3) {
                        LOG_INF("prune experts: step=%d sample=%s\n",
                                i, format_expert_sample(token_experts, 32).c_str());
                    }
                }

                if (reject_conf || reject_score || reject_budget || reject_depth || reject_nodes) {
                    pprof.stopped++;
                    if (!drafts[0].tokens.empty()) {
                        drafts[0].tokens.pop_back();
                    }
                    if (!drafts[0].probs.empty()) {
                        drafts[0].probs.pop_back();
                    }
                    if (!drafts[0].dists.empty()) {
                        drafts[0].dists.pop_back();
                    }
                    if (!drafts[0].i_batch_tgt.empty()) {
                        drafts[0].i_batch_tgt.pop_back();
                    }
                    if (batch_tgt.n_tokens > 1) {
                        --batch_tgt.n_tokens;
                    }
                    llama_memory_seq_rm(mem_dft, 0, n_past_dft + i, -1);
                    LOG_INF("prune stop: step=%d conf=%.4f offchip=%d new_offchip=%d delta_g=%.4f delta_c=%.4f score=%.4f budget=%.4f\n",
                            i, (double) conf, n_offchip_experts, n_new_experts, (double) delta_g, (double) delta_c, (double) score, (double) prune_budget_used);
                    break;
                }

                prune_budget_used += delta_c;
                ++prune_nodes;
                prune_seen_experts.insert(token_experts.begin(), token_experts.end());
            }

            if (batch_tgt.n_tokens > n_draft) {
                break;
            }
            if (stop_drafting) {
                break;
            }
        }

        // evaluate the target model on the drafted tokens
        {
            const int64_t t_memory_start = ggml_time_us();
            llama_memory_seq_keep(mem_tgt, 0);
            for (int s = 1; s < n_seq_dft; ++s) {
                llama_memory_seq_cp(mem_tgt, 0, s, -1, -1);
            }

            // LOG_DBG("target batch: %s\n", LOG_BATCH_TOSTR_PRETTY(ctx_tgt, batch_tgt).c_str());
            // ==========================================
            // [新增] 共享 KV Cache 的关键修正：逻辑回滚
            // ==========================================
            // Draft 刚刚写入了数据，导致 KV Cache 指针跑到了后面。
            // Target 需要从 n_past_tgt 开始验证（覆写）。
            // 所以我们需要把 n_past_tgt 之后的数据标记为“移除”（逻辑删除），
            // 这样 Target 就能从 n_past_tgt 开始写入了。
            
            // 判断是否共享了内存 (Self-Speculation)
            if (llama_get_memory(ctx_tgt) == llama_get_memory(ctx_dft)) {
                // 对所有 sequence (-1)，清除从 n_past_tgt 开始的数据
                llama_memory_seq_rm(mem_tgt, -1, n_past_tgt, -1);
            }
            prof.memory_us += ggml_time_us() - t_memory_start;
            // ==========================================
            llama_set_ssd_cuda_cache_mode(2);
            const int64_t t_decode_tgt_start = ggml_time_us();
            const bool reuse_verify = params.moe_reuse_runtime && params.moe_reuse_expert_cap != 0;
            llama_context_set_moe_reuse_verify(ctx_tgt, reuse_verify, reuse_verify);
            llama_decode(ctx_tgt, batch_tgt);
            llama_context_set_moe_reuse_verify(ctx_tgt, false, false);
            prof.target_decode_us += ggml_time_us() - t_decode_tgt_start;
            prof.target_decode_calls++;
            prof.target_decode_tokens += batch_tgt.n_tokens;
            llama_clear_ssd_cuda_cache();
            llama_set_ssd_cuda_cache_mode(0);
            llama_clear_ssd_cache_backend();
            ++n_past_tgt;
        }

        // the first token is always proposed by the target model before the speculation loop so we erase it here
        for (int s = 0; s < n_seq_dft; ++s) {
            if (!drafts[s].active) {
                continue;
            }

            drafts[s].tokens.erase(drafts[s].tokens.begin());
            // ================= [新增：同步删除概率] =================
            if (!drafts[s].probs.empty()) {
                drafts[s].probs.erase(drafts[s].probs.begin());
            }
            // ======================================================
            drafts[s].dists.erase(drafts[s].dists.begin());
        }
    }

    auto t_dec_end = ggml_time_us();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", n_draft);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);
    LOG_INF("\n");
    LOG_INF("spec profile:\n");
    LOG_INF("  draft sample:  %8.3f ms / %lld calls\n", prof.draft_sample_us / 1000.0, (long long) prof.draft_sample_calls);
    LOG_INF("  draft decode:  %8.3f ms / %lld calls / %lld tokens\n", prof.draft_decode_us / 1000.0, (long long) prof.draft_decode_calls, (long long) prof.draft_decode_tokens);
    LOG_INF("  target sample: %8.3f ms / %lld calls\n", prof.target_sample_us / 1000.0, (long long) prof.target_sample_calls);
    LOG_INF("  target verify: %8.3f ms / %lld calls / %lld tokens\n", prof.target_decode_us / 1000.0, (long long) prof.target_decode_calls, (long long) prof.target_decode_tokens);
    LOG_INF("  memory ops:    %8.3f ms\n", prof.memory_us / 1000.0);

    if (params.speculative.prune != 0) {
        const double denom = pprof.steps > 0 ? (double) pprof.steps : 1.0;
        LOG_INF("\n");
        LOG_INF("prune profile:\n");
        LOG_INF("  steps:         %lld stopped=%lld empty_topk=%lld\n",
                (long long) pprof.steps, (long long) pprof.stopped, (long long) pprof.empty_topk);
        LOG_INF("  experts/step:  total=%.3f resident=%.3f offchip=%.3f new_offchip=%.3f\n",
                pprof.total_token_experts / denom,
                pprof.total_resident_experts / denom,
                pprof.total_offchip_experts / denom,
                pprof.total_new_offchip_experts / denom);
        LOG_INF("  cost sums:     delta_g=%.3f delta_c=%.3f decode_ms=%.3f\n",
                pprof.total_delta_g,
                pprof.total_delta_c,
                pprof.total_decode_ms);
        LOG_INF("  cost/step:     delta_g=%.3f delta_c=%.3f decode_ms=%.3f\n",
                pprof.total_delta_g / denom,
                pprof.total_delta_c / denom,
                pprof.total_decode_ms / denom);
    }

    LOG_INF("\n");
    LOG_INF("draft:\n\n");
    // TODO: print sampling/grammar timings for all drafts
    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    for (int s = 0; s < n_seq_dft; ++s) {
        common_sampler_free(drafts[s].smpl);
    }

    llama_batch_free(batch_dft);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
