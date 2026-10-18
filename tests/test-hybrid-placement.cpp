#include "llama-model-loader.h"

#include <cstdio>
#include <stdexcept>

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int main(int argc, char ** argv) try {
    check(argc == 2, "expected fixture path");
    std::vector<std::string> splits;
    llama_model_loader original(argv[1], splits, false, false, false, nullptr, nullptr, nullptr);
    llama_model_loader ssd(argv[1], splits, false, false, true, nullptr, nullptr, nullptr);
    uint64_t dense = 0, first_bundle = 0;
    size_t projections = 0;
    for (const auto & item : ssd.weights_map) {
        int layer = -1, expert = -1;
        char kind[32];
        const auto * tensor = item.second.tensor;
        const uint64_t padding = ggml_is_quantized(tensor->type) && tensor->ne[0] % 512
            ? ggml_row_size(tensor->type, 512 - tensor->ne[0] % 512) : 0;
        const uint64_t aligned = (ggml_nbytes(tensor) + padding + 511) & ~uint64_t(511);
        if (std::sscanf(item.first.c_str(), "blk.%d.ffn_%31[^.].%d.weight", &layer, kind, &expert) != 3) {
            dense += aligned;
            continue;
        }
        ++projections;
        const auto name = format("blk.%d.ffn_%s_exps.weight", layer, kind);
        const auto & packed = original.require_weight(name.c_str());
        check(item.second.idx == packed.idx, "expert alias lost shard index");
        check(item.second.offs == packed.offs + expert * packed.tensor->nb[2], "wrong expert byte offset");
        check(ggml_nbytes(item.second.tensor) == packed.tensor->nb[2], "wrong expert length");
        check(item.second.tensor->type == packed.tensor->type, "expert quantization changed");
        if (layer == 0 && expert == 0) first_bundle += aligned;
    }
    check(projections == 8 * 4 * 3, "missing packed expert aliases");
    bool rejected = false;
    try { ssd.configure_ssd_budget(dense - 1, false); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "budget smaller than non-routed weights accepted");
    ssd.configure_ssd_budget(dense + first_bundle, false);
    check(ssd.hot_expert_indices == std::set<std::pair<int, int>>{{0, 0}}, "budget did not select one whole expert");
    ssd.hot_expert_indices.insert({0, 1});
    rejected = false;
    try { ssd.configure_ssd_budget(dense + first_bundle, true); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "oversized hot list accepted");
    std::puts("PASS packed offsets/types, dense floor, whole-expert placement, oversized hot-list rejection");
    return 0;
} catch (const std::exception & error) {
    std::fprintf(stderr, "placement test: %s\n", error.what());
    return 1;
}
