// Gate 2: inspect executed graph shapes, including target invariance and all
// recurrent/full-attention/shared-expert paths. This does not enable speculation.
#include "../examples/speculative-next/next-state.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <string>

using next_spec::require;

struct audit {
    bool enabled = false;
    int width = 0;
    int64_t projected_elements = 0;
    std::set<int> recurrent, attention, shared;
    std::map<std::string, int64_t> projections;
};

static bool inspect(ggml_tensor * tensor, bool ask, void * data) {
    auto & seen = *static_cast<audit *>(data);
    if (!seen.enabled || !ask) return false;
    const std::string name = tensor->name;
    auto layer = [&](const char * prefix, std::set<int> & ids) {
        if (name.compare(0, std::strlen(prefix), prefix) == 0) {
            ids.insert(std::stoi(name.substr(std::strlen(prefix))));
        }
    };
    layer("linear_attn_out-", seen.recurrent);
    layer("attn_gated-", seen.attention);
    layer("ffn_shexp-", seen.shared);
    for (const char * prefix : {"ffn_moe_up-", "ffn_moe_gate-", "ffn_moe_down-"}) {
        if (name.compare(0, std::strlen(prefix), prefix) == 0) {
            require(tensor->ne[1] == seen.width, "routed projection uses an unexpected expert count");
            seen.projections[name] = ggml_nelements(tensor);
        }
    }
    return false; // Metadata-only audit: no extra device synchronization/copy.
}

int main(int argc, char ** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s model [gpu-layers]\n", argv[0]); return 1; }
    try {
        next_spec::disable_graphs();
        llama_backend_init();
        ggml_backend_load_all();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = argc > 2 ? std::stoi(argv[2]) : 0;
        ggml_backend_dev_t cpu_only[] = {nullptr};
        if (!mp.n_gpu_layers) mp.devices = cpu_only;
        mp.moe_reuse_strength = 0;
        mp.moe_reuse_expert_cap = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model loading failed");
        char value[64] = {};
        llama_model_meta_val_str(model.get(), "general.architecture", value, sizeof(value));
        require(std::string(value) == "qwen35moe", "expected a Qwen3.5-MoE model");
        llama_model_meta_val_str(model.get(), "qwen35moe.expert_used_count", value, sizeof(value));
        const int full = std::stoi(value);
        const int layers = llama_model_n_layer(model.get());
        require(full >= 2, "test requires at least two active routed experts");
        auto run = [&](bool draft, int topk, audit & seen) {
            auto cp = llama_context_default_params();
            cp.n_ctx = 128; cp.n_batch = 32; cp.n_ubatch = 32; cp.n_seq_max = 1;
            cp.n_threads = cp.n_threads_batch = 4;
            cp.op_offload = mp.n_gpu_layers > 0;
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.cb_eval = inspect; cp.cb_eval_user_data = &seen;
            std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
                llama_init_from_model(model.get(), cp), llama_free);
            require(bool(ctx), "context creation failed");
            llama_context_set_draft_context(ctx.get(), draft);
            llama_context_set_draft_expert_topk(ctx.get(), topk);
            seen.width = draft ? std::min(topk, full) : full;
            seen.enabled = true;
            next_spec::decode(ctx.get(), {4, 8, 15}, 0);
            llama_synchronize(ctx.get());
            seen.enabled = false;
            require(int(seen.recurrent.size() + seen.attention.size()) == layers, "a recurrent/attention layer was omitted");
            require(int(seen.shared.size()) == layers, "a shared-expert path was omitted");
            require(int(seen.projections.size()) == 3 * layers, "missing routed up/gate/down projections");
            for (const auto & item : seen.projections) seen.projected_elements += item.second;
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            require(logits != nullptr, "missing draft/target logits");
            for (int i = 0; i < n_vocab; ++i) require(std::isfinite(logits[i]), "non-finite draft/target logits");
            return std::vector<float>(logits, logits + n_vocab);
        };
        audit target, unchanged;
        const auto reference = run(false, full, target);
        const auto ignored_override = run(false, 1, unchanged);
        require(reference == ignored_override, "draft top-k changed target logits");
        for (int k : std::set<int>{1, std::min(2, full), full}) {
            audit draft;
            run(true, k, draft);
            require(draft.recurrent == target.recurrent && draft.attention == target.attention &&
                    draft.shared == target.shared, "draft changed preserved layer paths");
            require(draft.projected_elements * full == target.projected_elements * k,
                    "routed projection work did not decrease with draft top-k");
            std::fprintf(stderr, "QWEN35_EXPERT_AUDIT topk=%d target_topk=%d layers=%d recurrent=%zu attention=%zu shared=%zu routed_elements=%lld target_elements=%lld\n",
                         k, full, layers, draft.recurrent.size(), draft.attention.size(), draft.shared.size(),
                         (long long) draft.projected_elements, (long long) target.projected_elements);
        }
        std::fprintf(stderr, "PASS expert-only graph audit and target invariance\n");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    return 0;
}
