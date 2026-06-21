#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <vector>

inline std::set<int> perturb_expert_ids(
        const std::set<int> & experts,
        float noise_rate,
        int expert_count,
        uint64_t seed) {
    if (experts.empty() || noise_rate <= 0.0f || expert_count <= 0) {
        return experts;
    }

    std::vector<int> original(experts.begin(), experts.end());
    std::vector<int> positions(original.size());
    std::iota(positions.begin(), positions.end(), 0);
    std::mt19937_64 rng(seed);
    std::shuffle(positions.begin(), positions.end(), rng);

    const size_t replace_count = std::min(
        original.size(),
        static_cast<size_t>(std::llround(noise_rate * original.size())));
    std::set<int> result = experts;
    std::vector<int> available;
    for (int expert = 0; expert < expert_count; ++expert) {
        if (experts.count(expert) == 0) {
            available.push_back(expert);
        }
    }
    std::shuffle(available.begin(), available.end(), rng);

    const size_t actual_count = std::min(replace_count, available.size());
    for (size_t i = 0; i < actual_count; ++i) {
        result.erase(original[positions[i]]);
        result.insert(available[i]);
    }
    return result;
}

inline std::vector<std::set<int>> perturb_layered_expert_ids(
        const std::vector<std::set<int>> & layers,
        float noise_rate,
        int expert_count,
        uint64_t seed) {
    if (noise_rate <= 0.0f || expert_count <= 0) {
        return layers;
    }

    struct Position {
        size_t layer;
        int expert;
    };
    std::vector<Position> positions;
    for (size_t layer = 0; layer < layers.size(); ++layer) {
        for (const int expert : layers[layer]) {
            positions.push_back({layer, expert});
        }
    }
    if (positions.empty()) {
        return layers;
    }

    std::mt19937_64 rng(seed);
    std::shuffle(positions.begin(), positions.end(), rng);
    const size_t replace_count = std::min(
        positions.size(),
        static_cast<size_t>(std::llround(noise_rate * positions.size())));

    auto result = layers;
    for (size_t i = 0; i < replace_count; ++i) {
        const auto & pos = positions[i];
        std::vector<int> available;
        for (int expert = 0; expert < expert_count; ++expert) {
            if (result[pos.layer].count(expert) == 0) {
                available.push_back(expert);
            }
        }
        if (available.empty()) {
            continue;
        }
        std::uniform_int_distribution<size_t> pick(0, available.size() - 1);
        result[pos.layer].erase(pos.expert);
        result[pos.layer].insert(available[pick(rng)]);
    }
    return result;
}
