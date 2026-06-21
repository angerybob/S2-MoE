#include "../prune_routing_noise.h"

#include <cassert>
#include <set>
#include <vector>

int main() {
    std::set<int> experts = {1, 3, 5, 7};

    const auto unchanged = perturb_expert_ids(experts, 0.0f, 16, 123);
    assert(unchanged == experts);

    const auto perturbed = perturb_expert_ids(experts, 0.5f, 16, 123);
    assert(perturbed.size() == experts.size());
    assert(perturbed != experts);
    for (const int expert : perturbed) {
        assert(expert >= 0 && expert < 16);
    }

    const auto repeated = perturb_expert_ids(experts, 0.5f, 16, 123);
    assert(repeated == perturbed);

    const auto fully_perturbed = perturb_expert_ids(experts, 1.0f, 16, 456);
    assert(fully_perturbed.size() == experts.size());
    for (const int expert : fully_perturbed) {
        assert(experts.count(expert) == 0);
    }

    const std::vector<std::set<int>> layered = {{1}, {3}, {5}, {7}};
    const auto layered_perturbed =
        perturb_layered_expert_ids(layered, 0.25f, 16, 789);
    assert(layered_perturbed.size() == layered.size());
    int changed = 0;
    for (size_t layer = 0; layer < layered.size(); ++layer) {
        assert(layered_perturbed[layer].size() == layered[layer].size());
        changed += layered_perturbed[layer] != layered[layer];
    }
    assert(changed == 1);

    return 0;
}
