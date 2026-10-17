#include "../examples/speculative-next/next-state.h"
#include "ggml-backend.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"

#include <cstdio>
#include <memory>

using next_spec::require;

static std::vector<float> logits(llama_context * ctx, int index = -1) {
    const int size = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    const float * data = llama_get_logits_ith(ctx, index);
    return {data, data + size};
}

static void compare(const std::vector<float> & a, const std::vector<float> & b, const char * label) {
    require(a.size() == b.size(), "comparison shape mismatch");
    for (size_t i = 0; i < a.size(); ++i) {
        const float tolerance = 2e-4f + 2e-3f * std::max(std::abs(a[i]), std::abs(b[i]));
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || std::abs(a[i] - b[i]) > tolerance) {
            std::fprintf(stderr, "%s mismatch at %zu: %.9g vs %.9g\n", label, i, a[i], b[i]);
            throw std::runtime_error(label);
        }
    }
}

static std::vector<float> recurrent(llama_context * ctx) {
    llama_synchronize(ctx);
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(llama_get_memory(ctx));
    require(hybrid != nullptr, "expected hybrid memory");
    auto * mem = hybrid->get_mem_recr();
    const int tail = mem->cells.at(0).tail;
    require(tail >= 0, "missing recurrent tail");
    std::vector<float> values;
    for (const auto * layers : {&mem->r_l, &mem->s_l}) {
        for (auto * tensor : *layers) {
            if (!tensor) continue;
            require(tensor->type == GGML_TYPE_F32, "test expects F32 recurrent states");
            const size_t bytes = ggml_nbytes(tensor) / mem->cells.size();
            const size_t offset = values.size();
            values.resize(offset + bytes / sizeof(float));
            ggml_backend_tensor_get(tensor, values.data() + offset, tail * bytes, bytes);
        }
    }
    return values;
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model.gguf [gpu-layers]\n", argv[0]); return 1; }
    try {
        next_spec::disable_graphs();
        llama_backend_init();
        ggml_backend_load_all();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = argc > 2 ? std::stoi(argv[2]) : 0;
        ggml_backend_dev_t cpu_only[] = {nullptr};
        if (mp.n_gpu_layers == 0) mp.devices = cpu_only;
        mp.moe_reuse_strength = 0;
        mp.moe_reuse_expert_cap = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(model != nullptr, "model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 128; cp.n_batch = 64; cp.n_ubatch = 32; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 2;
        cp.op_offload = mp.n_gpu_layers > 0;
        cp.kv_unified = true;
        // Isolate state/history correctness from F16 cache rounding near ties.
        // The executable-level AR test separately covers the default F16 cache.
        cp.type_k = cp.type_v = GGML_TYPE_F32;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        auto make_context = [&]() {
            std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
            require(ctx != nullptr, "context creation failed");
            return ctx;
        };
        auto target = make_context();
        auto reference = make_context();
        auto draft = make_context();
        llama_context_set_draft_context(draft.get(), true);
        llama_context_set_draft_expert_topk(draft.get(), 1);
        llama_share_kv_cache(draft.get(), target.get());
        auto * mem = llama_get_memory(target.get());
        require(!llama_memory_history_reserve(mem, 0), "accepted zero capacity");
        require(!llama_memory_history_reserve(mem, 33), "accepted excessive capacity");
        require(llama_memory_history_reserve(mem, 32), "reserve failed");
        const std::vector<llama_token> prompt{4, 8, 15};
        auto reset = [&](llama_context * ctx) {
            llama_memory_clear(llama_get_memory(ctx), true);
            for (size_t i = 0; i < prompt.size(); ++i) next_spec::decode(ctx, {prompt[i]}, i);
        };
        for (int n : {1, 2, 4, 9, 32}) {
            std::vector<llama_token> verification;
            for (int i = 0; i < n; ++i) verification.push_back(20 + i);
            for (int prefix = 1; prefix <= n; ++prefix) {
                reset(target.get());
                reset(reference.get());
                const next_spec::snapshot saved(target.get());
                const auto original = recurrent(target.get());
                next_spec::decode(draft.get(), {72}, prompt.size());
                saved.restore(target.get());
                compare(original, recurrent(target.get()), "snapshot after sparse draft");
                require(llama_memory_history_begin(mem, 0, prompt.size(), n), "begin failed");
                next_spec::decode(target.get(), verification, prompt.size(), true);
                require(llama_memory_history_finish(target.get()), "finish failed");
                require(!llama_memory_history_commit(target.get(), 0, 0, true), "accepted zero prefix");
                require(!llama_memory_history_commit(target.get(), 0, n + 1, true), "accepted excessive prefix");
                for (int i = 0; i < prefix; ++i) {
                    next_spec::decode(reference.get(), {verification[i]}, prompt.size() + i);
                    compare(logits(target.get(), i), logits(reference.get()), "batch vs AR logits");
                }
                require(llama_memory_history_commit(target.get(), 0, prefix, true), "commit failed");
                require(llama_memory_seq_pos_max(mem, 0) == int(prompt.size()) + prefix - 1, "bad committed position");
                compare(recurrent(target.get()), recurrent(reference.get()), "committed recurrent state");
                require(!llama_memory_history_commit(target.get(), 0, prefix, true), "history reused after commit");
                for (int step = 0; step < 3; ++step) {
                    const llama_token correction = 90 + step;
                    next_spec::decode(target.get(), {correction}, prompt.size() + prefix + step);
                    next_spec::decode(reference.get(), {correction}, prompt.size() + prefix + step);
                    compare(logits(target.get()), logits(reference.get()), "continuation logits");
                    compare(recurrent(target.get()), recurrent(reference.get()), "continuation state");
                }
            }
            std::fprintf(stderr, "PASS verification=%d all prefixes, root-only/partial/full, correction continuation\n", n);
        }
        reset(target.get());
        require(!llama_memory_history_begin(mem, 1, 3, 2), "accepted nonzero sequence");
        require(!llama_memory_history_begin(mem, 0, 4, 2), "accepted discontinuous history");
        require(!llama_memory_history_begin(mem, 0, 3, 0), "accepted empty history");
        require(!llama_memory_history_finish(target.get()), "finished without capture");
        cp.n_seq_max = 2;
        auto multi = make_context();
        require(!llama_memory_history_reserve(llama_get_memory(multi.get()), 4), "accepted multiple sequences");
        cp.n_seq_max = 1; cp.n_ubatch = 2;
        auto split = make_context();
        reset(split.get());
        require(llama_memory_history_reserve(llama_get_memory(split.get()), 4), "split reserve failed");
        require(llama_memory_history_begin(llama_get_memory(split.get()), 0, 3, 4), "split begin failed");
        bool rejected = false;
        try { next_spec::decode(split.get(), {20, 21, 22, 23}, 3, true); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "silently captured split verification");
        cp.n_ubatch = 64;
        auto long_batch = make_context();
        auto long_serial = make_context();
        const std::vector<llama_token> long_prompt(64, 42);
        next_spec::decode(long_batch.get(), long_prompt, 0);
        for (int i = 0; i < 64; ++i) next_spec::decode(long_serial.get(), {42}, i);
        compare(logits(long_batch.get()), logits(long_serial.get()), "64-token prefill logits");
        compare(recurrent(long_batch.get()), recurrent(long_serial.get()), "64-token prefill state");
        std::fprintf(stderr, "PASS 64-token prefill vs serial AR\n");
        std::fprintf(stderr, "PASS capacity, lifecycle, sequence, position, and split-ubatch rejection\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
