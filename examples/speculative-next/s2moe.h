#pragma once

#include "next-state.h"
#include "llama-model.h"

#include <map>
#include <set>
#include <string>

namespace next_spec {

struct prune_config {
    bool enabled = false;
    int max_depth = 31, max_nodes = 31;
    double budget_ms = 0, tpot_ms = 100, bandwidth = 1024 * 1024;
    double score_threshold = 1, confidence_threshold = 0, lambda = 1, epsilon = 1e-6;
};

// The policy uses the same expected-gain / incremental expert-I/O-cost
// criterion as the ordinary S2-MoE decoder. Hybrid state is handled separately
// by the driver so pruning cannot discard a live recurrent-state dependency.
class prune_policy {
    const prune_config & config;
    const llama_model & model;
    std::map<std::pair<int, int>, std::pair<uint64_t, bool>> experts;
public:
    struct path_state {
        std::set<std::pair<int, int>> seen;
        double probability = 1, cost_ms = 0;
    };
    path_state path;
    int kept = 0, rejected = 0;
    explicit prune_policy(const prune_config & config, const llama_model & model) : config(config), model(model) {
        for (const auto & item : model.get_ssd_experts()) {
            int layer, expert;
            char projection[32];
            if (std::sscanf(item.first.c_str(), "blk.%d.ffn_%31[^.].%d.weight", &layer, projection, &expert) != 3) {
                throw std::runtime_error("invalid expert registry name");
            }
            auto & bundle = experts[{layer, expert}];
            bundle.first += item.second.size;
            bundle.second = item.second.data_ptr != nullptr;
        }
    }
    void reset() { path = {}; kept = rejected = 0; }
    bool select(llama_context * draft, int depth, llama_token proposed, path_state & state) {
        if (!config.enabled) return true;
        const auto * logits = llama_get_logits_ith(draft, -1);
        const auto nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(draft)));
        const double maximum = *std::max_element(logits, logits + nv);
        double denominator = 0;
        for (int i = 0; i < nv; ++i) denominator += std::exp(logits[i] - maximum);
        const double confidence = std::exp(logits[proposed] - maximum) / denominator;
        const double path_probability = state.probability * std::pow(confidence, config.lambda);
        std::vector<llama_moe_topk_layer> routing(model.hparams.n_layer);
        int count = 0;
        require(llama_get_last_moe_topk(draft, 0, routing.data(), routing.size(), &count) && count > 0,
                "pruning requires captured MoE routing");
        auto next_seen = state.seen;
        uint64_t new_bytes = 0;
        for (int i = 0; i < count; ++i) {
            const auto & layer = routing[i];
            for (int j = 0; j < std::min(layer.n, int(model.hparams.n_expert_used)); ++j) {
                const auto key = std::make_pair(layer.layer, layer.expert_id[j]);
                if (!next_seen.insert(key).second) continue;
                const auto it = experts.find(key);
                // Fully resident model: expert I/O has no marginal SSD cost.
                if (it != experts.end() && !it->second.second) new_bytes += it->second.first;
            }
        }
        const double increment = new_bytes / config.bandwidth;
        const double score = path_probability * config.tpot_ms / (increment + config.epsilon);
        const bool keep = depth <= config.max_depth && kept < config.max_nodes &&
            (!config.budget_ms || state.cost_ms + increment <= config.budget_ms) &&
            confidence >= config.confidence_threshold && (depth == 1 || score >= config.score_threshold);
        if (!keep) { ++rejected; return false; }
        ++kept;
        state.seen = std::move(next_seen);
        state.probability = path_probability;
        state.cost_ms += increment;
        return true;
    }
    bool select(llama_context * draft, int depth, llama_token proposed) { return select(draft, depth, proposed, path); }
};

} // namespace next_spec
