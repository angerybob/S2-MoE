// Same-prefix diagnostic: separates batched arithmetic, history capture,
// prefix commit, and subsequent continuation. Uses forced AR inputs throughout.
#include "../examples/speculative-next/next-state.h"
#include "ggml-backend.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <string>

using json = nlohmann::ordered_json;
using next_spec::require;
using state_values = std::map<std::string, std::vector<float>>;

static state_values states(llama_context * ctx) {
    llama_synchronize(ctx);
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    require(hybrid != nullptr, "expected hybrid memory");
    auto * mem = hybrid->get_mem_recr();
    const int tail = mem->cells.at(0).tail;
    require(tail >= 0, "missing recurrent tail");
    state_values out;
    for (int kind = 0; kind < 2; ++kind) {
        const auto & layers = kind ? mem->s_l : mem->r_l;
        for (size_t il = 0; il < layers.size(); ++il) {
            auto * tensor = layers[il];
            if (!tensor) continue;
            require(tensor->type == GGML_TYPE_F32, "expected F32 recurrent state");
            const size_t bytes = ggml_nbytes(tensor) / mem->cells.size();
            auto & values = out[(kind ? "s-" : "r-") + std::to_string(il)];
            values.resize(bytes / sizeof(float));
            ggml_backend_tensor_get(tensor, values.data(), tail * bytes, bytes);
        }
    }
    return out;
}

static std::vector<float> logits(llama_context * ctx, int index = -1) {
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const auto * data = llama_get_logits_ith(ctx, index);
    return {data, data + nv};
}

static json compare(const std::vector<float> & reference, const std::vector<float> & candidate) {
    require(reference.size() == candidate.size(), "comparison shape mismatch");
    double square = 0, reference_square = 0;
    float maximum = 0;
    size_t different = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        require(std::isfinite(reference[i]) && std::isfinite(candidate[i]), "non-finite comparison");
        const double delta = double(reference[i]) - candidate[i];
        square += delta * delta;
        reference_square += double(reference[i]) * reference[i];
        maximum = std::max(maximum, float(std::abs(delta)));
        different += reference[i] != candidate[i];
    }
    return {{"max_abs", maximum}, {"rms", std::sqrt(square / reference.size())},
            {"relative_l2", std::sqrt(square / std::max(reference_square, 1e-30))}, {"different", different}};
}

static json compare_states(const state_values & reference, const state_values & candidate) {
    json result = json::object();
    for (const auto & item : reference) result[item.first] = compare(item.second, candidate.at(item.first));
    return result;
}

int main(int argc, char ** argv) try {
    require(argc == 5, "usage: diagnose-qwen-hybrid model ar-tokens.json prompt report.json");
    next_spec::disable_graphs();
    next_spec::enable_q8_cuda_ar_compatibility();
    llama_backend_init();
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    mp.use_mmap = false;
    mp.moe_reuse_strength = 0;
    mp.moe_reuse_expert_cap = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
    require(bool(model), "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 128; cp.n_batch = 64; cp.n_ubatch = 32; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 8; cp.kv_unified = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
    require(bool(ctx), "context creation failed");
    auto * memory = llama_get_memory(ctx.get());
    require(llama_memory_history_reserve(memory, 2), "history reserve failed");
    std::ifstream source(argv[2]);
    const std::vector<llama_token> ar = json::parse(source).get<std::vector<llama_token>>();
    require(ar.size() >= 20, "need at least twenty reference tokens");
    const std::string text = argv[3];
    std::vector<llama_token> prompt(text.size() + 32);
    int count = llama_tokenize(llama_model_get_vocab(model.get()), text.c_str(), text.size(), prompt.data(), prompt.size(), true, true);
    require(count > 0, "tokenization failed");
    prompt.resize(count);
    json report = {{"prompt_tokens", prompt}, {"probes", json::array()}};
    for (int offset : {0, 8, 16}) {
        llama_memory_clear(memory, true);
        llama_ssd_profile_set_phase(0);
        next_spec::decode(ctx.get(), prompt, 0);
        llama_ssd_profile_set_phase(2);
        for (int i = 0; i < offset; ++i) next_spec::decode(ctx.get(), {ar[i]}, count + i);
        json item = {{"input_offset", offset}, {"prefix_greedy", next_spec::greedy(ctx.get())}, {"expected_root", ar[offset]}};
        const next_spec::snapshot prefix(ctx.get());
        const llama_pos pos = count + offset;
        next_spec::decode(ctx.get(), {ar[offset]}, pos, true);
        const auto one_logits = logits(ctx.get());
        const auto one_state = states(ctx.get());
        next_spec::decode(ctx.get(), {ar[offset + 1]}, pos + 1, true);
        const auto two_logits = logits(ctx.get());
        const auto two_state = states(ctx.get());
        for (bool history : {false, true}) {
            prefix.restore(ctx.get());
            if (history) require(llama_memory_history_begin(memory, 0, pos, 2), "begin failed");
            next_spec::decode(ctx.get(), {ar[offset], ar[offset + 1]}, pos, true);
            if (history) require(llama_memory_history_finish(ctx.get()), "finish failed");
            const auto key = history ? "capture" : "no_capture";
            item[key] = {{"row0_logits", compare(one_logits, logits(ctx.get(), 0))},
                         {"row1_logits", compare(two_logits, logits(ctx.get(), 1))},
                         {"final_states", compare_states(two_state, states(ctx.get()))}};
            if (history) {
                require(llama_memory_history_commit(ctx.get(), 0, 1, true), "commit failed");
                item[key]["committed_row0_states"] = compare_states(one_state, states(ctx.get()));
                next_spec::decode(ctx.get(), {ar[offset + 1]}, pos + 1, true);
                item[key]["continuation_logits"] = compare(two_logits, logits(ctx.get()));
                item[key]["continuation_states"] = compare_states(two_state, states(ctx.get()));
            }
        }
        report["probes"].push_back(item);
        std::ofstream out(argv[4]);
        out << report.dump(2) << '\n';
        std::fprintf(stderr, "PROBE offset=%d prefix=%d expected=%d no_capture=%s capture=%s continuation=%s\n",
            offset, int(item["prefix_greedy"]), int(item["expected_root"]),
            item["no_capture"]["row1_logits"].dump().c_str(), item["capture"]["row1_logits"].dump().c_str(),
            item["capture"]["continuation_logits"].dump().c_str());
    }
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
