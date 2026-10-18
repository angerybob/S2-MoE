#include "next-state.h"
#include "ggml-backend.h"
#include "s2moe.h"
#include "hybrid-driver.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

struct options {
    std::string model, prompt = "The capital of France is", hot, tokens_file, timings_file, benchmark_plan;
    std::string draft_model;
    int predict = 32, draft = 8, experts = 2, gpu_layers = 99, context = 512, ubatch = 32, threads = 8;
    int probe_batch2_at = -1;
    bool ssd = false, ar = false, audit = false, ignore_eos = false, quiet = false, trace_logits = false;
    uint64_t memory_mib = 0, reserve_mib = 2048;
    float reuse_strength = 0;
    int reuse_cap = 0;
    int branches = 1;
    bool reuse_runtime = false, direct_io = true;
    bool no_mmap = false;
    next_spec::prune_config prune;
};

// Opt-in wall-clock profiling. Synchronization at phase boundaries includes
// backend completion; verification includes writing recurrent-state history.
struct phase_timer {
    using clock = std::chrono::steady_clock;
    bool enabled;
    double & elapsed;
    clock::time_point start;
    phase_timer(bool enabled, double & elapsed) : enabled(enabled), elapsed(elapsed) {
        if (enabled) start = clock::now();
    }
    ~phase_timer() {
        if (enabled) elapsed += std::chrono::duration<double, std::milli>(clock::now() - start).count();
    }
};

struct hybrid_tree {
    struct node { llama_token token; std::vector<int> children; };
    std::vector<node> nodes;
    std::vector<int> path(int first) const {
        std::vector<int> result;
        for (;;) {
            result.push_back(first);
            if (nodes[first].children.empty()) return result;
            first = nodes[first].children.front();
        }
    }
    int child(int parent, llama_token token) const {
        for (int i : nodes[parent].children) if (nodes[i].token == token) return i;
        return -1;
    }
};

static hybrid_tree build_tree(const options & o, llama_context * draft, llama_token root, llama_pos pos,
        int depth_limit, next_spec::prune_policy & pruning, double & fork_save_ms, double & fork_restore_ms) {
    hybrid_tree tree;
    tree.nodes.push_back({root, {}});
    // DFS restores only an ancestor's state: its attention prefix has never
    // been overwritten. A partial snapshot is NOT safe for arbitrary best-first
    // branch switching, which could retain another branch's attention KV.
    const int node_limit = o.prune.max_nodes;
    std::function<void(int, int, next_spec::prune_policy::path_state)> expand;
    expand = [&](int index, int depth, next_spec::prune_policy::path_state state) {
        if (depth >= depth_limit || int(tree.nodes.size()) > node_limit) return;
        next_spec::decode(draft, {tree.nodes[index].token}, pos + depth);
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(draft)));
        const float * logits = llama_get_logits_ith(draft, -1);
        next_spec::greedy(draft); // validate every logit before ordering branches
        // Copy choices before recursive decoding overwrites the logits buffer.
        std::vector<llama_token> choices(nv);
        for (int i = 0; i < nv; ++i) choices[i] = i;
        const int width = std::min(o.branches, nv);
        std::partial_sort(choices.begin(), choices.begin() + width, choices.end(), [&](int a, int b) {
            return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
        });
        choices.resize(width);
        std::vector<std::pair<llama_token, next_spec::prune_policy::path_state>> selected;
        for (llama_token token : choices) {
            if (int(tree.nodes.size() + selected.size()) > node_limit) break;
            auto branch = state;
            if (pruning.select(draft, depth + 1, token, branch)) selected.emplace_back(token, std::move(branch));
        }
        // Add siblings before expanding the first child, so a deep primary
        // branch cannot consume the node budget reserved for its alternatives.
        std::vector<int> children;
        for (const auto & choice : selected) {
            children.push_back(tree.nodes.size());
            tree.nodes.push_back({choice.first, {}});
        }
        tree.nodes[index].children = children;
        if (depth + 1 >= depth_limit || children.empty()) return;
        std::optional<next_spec::snapshot> parent;
        if (children.size() > 1) {
            phase_timer timer(!o.timings_file.empty(), fork_save_ms);
            parent.emplace(draft);
        }
        for (size_t i = 0; i < children.size(); ++i) {
            if (int(tree.nodes.size()) > node_limit) break;
            if (i) {
                phase_timer timer(!o.timings_file.empty(), fork_restore_ms);
                parent->restore(draft);
            }
            expand(children[i], depth + 1, std::move(selected[i].second));
        }
    };
    expand(0, 0, {});
    return tree;
}

static void usage(const char * exe) {
    std::fprintf(stderr,
#ifdef LLAMA_HYBRID_DRIVER
        "Experimental Qwen hybrid S2-MoE (Next / Qwen3.5-MoE)\n"
#elif defined(LLAMA_EXPERIMENTAL_QWEN35)
        "Experimental greedy Qwen3.5-MoE expert-only self-speculation\n"
#else
        "Experimental greedy Qwen3-Next expert-only self-speculation\n"
#endif
        "Usage: %s -m model.gguf [-p prompt] [-n 32] [-ngl 99] [-c 512] [-ub 32]\n"
        "       [--draft-max 8] [--draft-expert-topk 2] [--ssd-moe] [--hot-experts file]\n"
        "       [--memory-budget-mib N] [--runtime-reserve-mib 2048] [--ssd-buffered-io]\n"
        "       [--prune 1] [--prune-max-depth 31] [--prune-max-nodes 31]\n"
        "       [--draft-branches 1] (bounded tree; staged path verification for hybrid state)\n"
        "       [--prune-budget 0] [--prune-tpot 100] [--prune-bandwidth 1]\n"
        "       [--prune-score-thresh 1] [--prune-conf-thresh 0] [--prune-lambda 1]\n"
        "       [--moe-reuse-strength 0] [--moe-reuse-expert-cap 0] [--moe-reuse-runtime]\n"
        "       [--audit-state-copies] [--ar] [--tokens-file file] [--ignore-eos] [-t 8]\n"
        "       [--timings-file file] [--quiet]\n"
        "       [--benchmark-plan file] (reuse model, fresh contexts for each case)\n"
        "       [--trace-logits] (diagnostic target top-two logits; affects timings)\n"
        "       [--probe-batch2-at output-index] (AR-only diagnostic; affects timings)\n"
        "One request, shared hybrid memory, raw prompt, greedy sampling only.\n"
        "All recurrent/full-attention layers and shared experts remain active.\n"
#if defined(LLAMA_EXPERIMENTAL_QWEN35) || defined(LLAMA_HYBRID_DRIVER)
        "CUDA graphs and graph reuse are disabled. Vision/MTP are unsupported.\n", exe);
#else
        "CUDA graphs and graph reuse are disabled automatically. Qwen3.5 is unsupported.\n", exe);
#endif
}

static options parse(int argc, char ** argv) {
    options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            next_spec::require(i + 1 < argc, "missing option value");
            return argv[++i];
        };
        auto integer = [&]() {
            const std::string text = value();
            size_t used = 0;
            const int result = std::stoi(text, &used);
            next_spec::require(used == text.size(), "invalid integer option");
            return result;
        };
        auto real = [&]() {
            const auto text = value();
            size_t used = 0;
            double result = std::stod(text, &used);
            next_spec::require(used == text.size() && std::isfinite(result), "invalid real option");
            return result;
        };
        if (arg == "-m" || arg == "--model") o.model = value();
        else if (arg == "--model-draft") o.draft_model = value();
        else if (arg == "-p" || arg == "--prompt") o.prompt = value();
        else if (arg == "-n" || arg == "--predict") o.predict = integer();
        else if (arg == "-ngl" || arg == "--gpu-layers") o.gpu_layers = integer();
        else if (arg == "-c" || arg == "--ctx-size") o.context = integer();
        else if (arg == "-ub" || arg == "--ubatch-size") o.ubatch = integer();
        else if (arg == "-t" || arg == "--threads") o.threads = integer();
        else if (arg == "--draft-max" || arg == "--draft") o.draft = integer();
        else if (arg == "--draft-branches" || arg == "--parallel" || arg == "-np") o.branches = integer();
        else if (arg == "--draft-expert-topk") o.experts = integer();
        else if (arg == "--hot-experts") o.hot = value();
        else if (arg == "--memory-budget-mib") { const int n = integer(); next_spec::require(n > 0, "memory budget must be positive"); o.memory_mib = n; }
        else if (arg == "--runtime-reserve-mib") { const int n = integer(); next_spec::require(n > 0, "runtime reserve must be positive"); o.reserve_mib = n; }
        else if (arg == "--ssd-buffered-io") o.direct_io = false;
        else if (arg == "--prune") { const int n = integer(); next_spec::require(n == 0 || n == 1, "--prune must be 0 or 1"); o.prune.enabled = n; }
        else if (arg == "--prune-max-depth") o.prune.max_depth = integer();
        else if (arg == "--prune-max-nodes") o.prune.max_nodes = integer();
        else if (arg == "--prune-budget") o.prune.budget_ms = real();
        else if (arg == "--prune-tpot") o.prune.tpot_ms = real();
        else if (arg == "--prune-bandwidth") o.prune.bandwidth = real() * 1024 * 1024;
        else if (arg == "--prune-score-thresh") o.prune.score_threshold = real();
        else if (arg == "--prune-conf-thresh") o.prune.confidence_threshold = real();
        else if (arg == "--prune-lambda") o.prune.lambda = real();
        else if (arg == "--prune-eps") o.prune.epsilon = real();
        else if (arg == "--moe-reuse-strength") o.reuse_strength = real();
        else if (arg == "--moe-reuse-expert-cap") o.reuse_cap = integer();
        else if (arg == "--moe-reuse-runtime") o.reuse_runtime = true;
        else if (arg == "--tokens-file") o.tokens_file = value();
        else if (arg == "--timings-file") o.timings_file = value();
        else if (arg == "--benchmark-plan") o.benchmark_plan = value();
        else if (arg == "--quiet") o.quiet = true;
        else if (arg == "--trace-logits") o.trace_logits = true;
        else if (arg == "--probe-batch2-at") o.probe_batch2_at = integer();
        else if (arg == "--ssd-moe") o.ssd = true;
        else if (arg == "--no-mmap") o.no_mmap = true;
        else if (arg == "--no-warmup" || arg == "-kvu") { /* hybrid contexts already use these policies */ }
        else if (arg == "--ar") o.ar = true;
        else if (arg == "--audit-state-copies") o.audit = true;
        else if (arg == "--ignore-eos") o.ignore_eos = true;
        else if (arg == "--draft-share-kv") { /* always shared in this entry point */ }
        else if (arg == "--temp") next_spec::require(real() == 0, "only --temp 0 is supported");
        else throw std::runtime_error("unsupported option: " + arg);
    }
    next_spec::require(!o.model.empty(), "-m is required");
    next_spec::require(o.draft_model.empty() || o.draft_model == o.model, "hybrid S2-MoE requires a shared self-draft model");
    next_spec::require(o.predict >= 0 && o.context > 0 && o.threads > 0, "invalid generation limits");
    next_spec::require(o.draft >= 0 && o.draft <= 31, "--draft-max must be in [0, 31]");
    next_spec::require(o.ubatch >= o.draft + 1 && o.ubatch <= 512,
                       "-ub must fit root + draft in one ubatch and be at most 512");
    next_spec::require(o.experts > 0, "--draft-expert-topk must be positive");
    next_spec::require(o.branches > 0 && o.branches <= 8, "--draft-branches must be in [1, 8]");
    next_spec::require(o.prune.max_depth >= 0 && o.prune.max_depth <= 31 && o.prune.max_nodes >= 0 && o.prune.max_nodes <= 512 &&
        o.prune.budget_ms >= 0 && o.prune.tpot_ms > 0 && o.prune.bandwidth > 0 && o.prune.score_threshold >= 0 &&
        o.prune.confidence_threshold >= 0 && o.prune.confidence_threshold <= 1 && o.prune.lambda >= 0 &&
        o.prune.epsilon > 0, "invalid pruning configuration");
    next_spec::require(o.reuse_strength >= 0 && o.reuse_cap >= 0, "invalid expert reuse configuration");
    next_spec::require(o.ssd || (o.hot.empty() && !o.memory_mib), "placement options require --ssd-moe");
    next_spec::require(!o.memory_mib || o.reserve_mib < o.memory_mib, "runtime reserve must be smaller than memory budget");
    next_spec::require(!o.ssd || o.gpu_layers >= 99, "hybrid SSD mode requires all compute layers on GPU (-ngl 99)");
    return o;
}

static int generate(const options & o, llama_model * model, double model_load_ms) {
    const bool profile = !o.timings_file.empty();
    const auto setup_start = phase_timer::clock::now();
    double prefill_ms = 0, draft_ms = 0, verify_ms = 0, snapshot_ms = 0;
    double restore_ms = 0, commit_ms = 0, ar_decode_ms = 0, sampling_ms = 0;
    double fork_save_ms = 0, fork_restore_ms = 0;
    auto cp = llama_context_default_params();
    cp.n_ctx = o.context;
    cp.n_batch = std::max(64, o.ubatch);
    cp.n_ubatch = o.ubatch;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = o.threads;
    cp.op_offload = o.gpu_layers > 0;
    cp.kv_unified = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    std::unique_ptr<llama_context, decltype(&llama_free)> target(
        llama_init_from_model(model, cp), llama_free);
    next_spec::require(target != nullptr, "target context creation failed");
    // Declaration order ensures the draft releases its borrowed memory first.
    std::unique_ptr<llama_context, decltype(&llama_free)> draft(nullptr, llama_free);
    if (!o.ar) {
        draft.reset(llama_init_from_model(model, cp));
        next_spec::require(draft != nullptr, "draft context creation failed");
        llama_context_set_draft_context(draft.get(), true);
        llama_context_set_draft_expert_topk(draft.get(), o.experts);
        llama_share_kv_cache(draft.get(), target.get());
        next_spec::require(llama_memory_history_reserve(llama_get_memory(target.get()), o.draft + 1),
                           "history reservation failed");
        if (o.prune.enabled) {
            llama_set_moe_topk(draft.get(), true);
            llama_context_set_moe_topk_log_k(draft.get(), model->hparams.n_expert_used);
        }
    }
    next_spec::prune_policy pruning(o.prune, *model);
    const auto memory = llama_get_buffer_usage(target.get());
    // Shared hybrid state is counted once. Include a host snapshot and SSD
    // staging (host + temporary device copy), plus allocator/metadata headroom.
    uint64_t snapshot_bytes = draft ? llama_state_seq_get_size_ext(target.get(), 0,
        LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) : 0;
    uint64_t runtime_bytes = 0;
    auto check_memory = [&]() {
        const auto active = llama_get_buffer_usage(target.get());
        const auto other = draft ? llama_get_buffer_usage(draft.get()) : llama_buffer_usage{};
        const uint64_t snapshots = o.branches > 1 ? 1 + std::min({o.draft, o.prune.max_nodes,
            o.prune.enabled ? o.prune.max_depth : o.draft}) : 1;
        runtime_bytes = std::max(runtime_bytes, active.state_bytes + active.compute_bytes + other.compute_bytes +
            snapshots * snapshot_bytes + (o.ssd ? uint64_t(128 * 1024 * 1024 + 8192) : uint64_t(0)) + uint64_t(64 * 1024 * 1024));
        if (o.memory_mib) {
            next_spec::require(runtime_bytes <= o.reserve_mib * 1024 * 1024,
                "runtime buffers exceed reserve; increase --runtime-reserve-mib (fewer experts will be resident)");
            next_spec::require(active.model_bytes + runtime_bytes <= o.memory_mib * 1024 * 1024,
                "allocated model/state/compute buffers exceed memory budget");
        }
    };
    check_memory();
    std::fprintf(stderr, "S2MOE_MEMORY model_bytes=%llu runtime_bytes=%llu reserve_mib=%llu budget_mib=%llu\n",
        (unsigned long long) memory.model_bytes, (unsigned long long) runtime_bytes,
        (unsigned long long) o.reserve_mib, (unsigned long long) o.memory_mib);
    const auto * vocab = llama_model_get_vocab(model);
    const auto target_greedy = [&](int index = -1) {
        const llama_token selected = next_spec::greedy(target.get(), index);
        if (o.trace_logits) {
            const float * logits = llama_get_logits_ith(target.get(), index);
            llama_token runner_up = selected == 0 ? 1 : 0;
            for (llama_token token = 0; token < llama_vocab_n_tokens(vocab); ++token) {
                if (token != selected && logits[token] > logits[runner_up]) runner_up = token;
            }
            std::fprintf(stderr, "NEXT_SPEC_LOGIT_TOP {\"selected\":%d,\"runner_up\":%d,\"score\":%.9g,\"runner_score\":%.9g,\"margin\":%.9g,\"row\":%d}\n",
                selected, runner_up, logits[selected], logits[runner_up], logits[selected] - logits[runner_up], index);
        }
        return selected;
    };
    std::vector<llama_token> prompt(o.prompt.size() + 16);
    int count = llama_tokenize(vocab, o.prompt.c_str(), o.prompt.size(), prompt.data(), prompt.size(), true, true);
    if (count < 0) {
        prompt.resize(-count);
        count = llama_tokenize(vocab, o.prompt.c_str(), o.prompt.size(), prompt.data(), prompt.size(), true, true);
    }
    next_spec::require(count > 0, "prompt must tokenize to at least one token");
    prompt.resize(count);
    next_spec::require(int64_t(count) + o.predict <= o.context, "prompt + output exceeds context capacity");
    const double setup_ms = std::chrono::duration<double, std::milli>(phase_timer::clock::now() - setup_start).count()
        + (o.benchmark_plan.empty() ? model_load_ms : 0);
    {
        phase_timer timer(profile, prefill_ms);
        llama_ssd_profile_reset();
        llama_ssd_profile_set_phase(0);
        for (int off = 0; off < count; off += o.ubatch) {
            next_spec::decode(target.get(), {prompt.begin() + off, prompt.begin() + std::min(count, off + o.ubatch)}, off);
        }
        if (profile) llama_synchronize(target.get());
    }
    snapshot_bytes = draft ? llama_state_seq_get_size_ext(target.get(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) : 0;
    check_memory();
    std::vector<llama_token> emitted;
    auto emit = [&](llama_token token) {
        if (!o.ignore_eos && llama_vocab_is_eog(vocab, token)) return false;
        emitted.push_back(token);
        if (o.quiet) return true;
        std::vector<char> piece(128);
        int size = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
        if (size < 0) {
            piece.resize(-size);
            size = llama_token_to_piece(vocab, token, piece.data(), piece.size(), 0, true);
        }
        next_spec::require(size >= 0, "token rendering failed");
        std::fwrite(piece.data(), 1, size, stdout);
        std::fflush(stdout);
        return true;
    };
    int pos = count, accepted_total = 0, proposed_total = 0, rounds = 0;
    int pruned = 0, tree_switches = 0, verification_batches = 0;
    llama_ssd_profile prefill_io{};
    llama_ssd_profile_snapshot(&prefill_io);
    llama_ssd_profile_reset();
    const auto generation_start = phase_timer::clock::now();
    llama_token root;
    {
        phase_timer timer(profile, sampling_ms);
        root = target_greedy();
    }
    int zero_rounds = 0, partial_rounds = 0, full_rounds = 0;
    bool stopped = false;
    while (int(emitted.size()) < o.predict && emit(root)) {
        if (int(emitted.size()) == o.predict) break;
        if (o.ar) {
            llama_ssd_profile_set_phase(2);
            if (o.probe_batch2_at == int(emitted.size())) {
                next_spec::require(o.ubatch >= 2, "batch probe requires ubatch >= 2");
                const next_spec::snapshot saved(target.get());
                next_spec::decode(target.get(), {root}, pos, true);
                const llama_token single_token = next_spec::greedy(target.get(), 0);
                const int n_vocab = llama_vocab_n_tokens(vocab);
                const float * one = llama_get_logits_ith(target.get(), 0);
                const std::vector<float> single(one, one + n_vocab);
                saved.restore(target.get());
                next_spec::decode(target.get(), {root, single_token}, pos, true);
                const llama_token batch_token = next_spec::greedy(target.get(), 0);
                const float * two = llama_get_logits_ith(target.get(), 0);
                double squared = 0;
                float max_abs = 0;
                for (int i = 0; i < n_vocab; ++i) {
                    const double delta = double(single[i]) - two[i];
                    squared += delta * delta;
                    max_abs = std::max(max_abs, float(std::abs(delta)));
                }
                std::fprintf(stderr, "NEXT_SPEC_BATCH_PROBE {\"output_index\":%d,\"single_token\":%d,\"batch_token\":%d,\"single_chosen_score\":%.9g,\"single_other_score\":%.9g,\"batch_chosen_score\":%.9g,\"batch_other_score\":%.9g,\"max_abs\":%.9g,\"rms\":%.9g}\n",
                    o.probe_batch2_at, single_token, batch_token,
                    single[single_token], single[batch_token], two[batch_token], two[single_token],
                    max_abs, std::sqrt(squared / n_vocab));
                saved.restore(target.get());
            }
            {
                phase_timer timer(profile, ar_decode_ms);
                next_spec::decode(target.get(), {root}, pos++);
                if (profile) llama_synchronize(target.get());
            }
            {
                phase_timer timer(profile, sampling_ms);
                root = target_greedy();
            }
            continue;
        }
        std::optional<next_spec::snapshot> prefix;
        {
            phase_timer timer(profile, snapshot_ms);
            prefix.emplace(target.get());
        }
        const int limit = std::min({o.draft, o.predict - int(emitted.size()),
            o.prune.enabled ? o.prune.max_depth : o.draft,
            o.prune.enabled ? o.prune.max_nodes : o.draft});
        hybrid_tree tree;
        pruning.reset();
        {
            phase_timer timer(profile, draft_ms);
            llama_ssd_profile_set_phase(1);
            tree = build_tree(o, draft.get(), root, pos, limit, pruning, fork_save_ms, fork_restore_ms);
        }
        pruned += pruning.rejected;
        const int budget = int(tree.nodes.size()) - 1;
        proposed_total += budget;
        {
            phase_timer timer(profile, restore_ms);
            prefix->restore(target.get());
        }
        int accepted = 0;
        int current = 0;
        bool reached_leaf = false;
        while (current >= 0) {
            const auto path = tree.path(current);
            std::vector<llama_token> verification;
            for (int index : path) verification.push_back(tree.nodes[index].token);
            {
                phase_timer timer(profile, verify_ms);
                llama_ssd_profile_set_phase(2);
                const bool reuse = o.reuse_cap > 0 && (o.reuse_runtime || o.reuse_strength > 0);
                llama_context_set_moe_reuse_verify(target.get(), reuse, o.reuse_runtime);
                next_spec::require(llama_memory_history_begin(llama_get_memory(target.get()), 0, pos, verification.size()),
                                   "could not begin verification history");
                next_spec::decode(target.get(), verification, pos, true);
                next_spec::require(llama_memory_history_finish(target.get()), "incomplete verification history");
                llama_context_set_moe_reuse_verify(target.get(), false, false);
                ++verification_batches;
            }
            int prefix_tokens = 0, alternate = -1;
            {
                phase_timer timer(profile, sampling_ms);
                for (size_t i = 0; i < path.size(); ++i) {
                    ++prefix_tokens;
                    root = target_greedy(i);
                    reached_leaf = tree.nodes[path[i]].children.empty();
                    const int matching = tree.child(path[i], root);
                    if (matching < 0) break;
                    if (!emit(root)) { stopped = true; break; }
                    ++accepted;
                    if (i + 1 == path.size() || path[i + 1] != matching) {
                        alternate = matching;
                        ++tree_switches;
                        break;
                    }
                }
            }
            {
                phase_timer timer(profile, commit_ms);
                next_spec::require(llama_memory_history_commit(target.get(), 0, prefix_tokens, o.audit),
                                   "accepted-prefix history commit failed");
                if (profile) llama_synchronize(target.get());
            }
            pos += prefix_tokens;
            if (stopped || int(emitted.size()) >= o.predict) break;
            current = alternate;
        }
        if (accepted == 0) ++zero_rounds;
        else if (reached_leaf) ++full_rounds;
        else ++partial_rounds;
        check_memory();
        accepted_total += accepted;
        ++rounds;
        if (stopped) break;
    }
    if (profile) llama_synchronize(target.get());
    const double generation_ms = std::chrono::duration<double, std::milli>(
        phase_timer::clock::now() - generation_start).count();
    if (!o.quiet) std::fputc('\n', stdout);
    if (profile) {
        llama_ssd_profile io{};
        llama_ssd_profile_snapshot(&io);
        std::ofstream out(o.timings_file);
        next_spec::require(out.good(), "cannot open timings output");
        out.precision(12);
        out << "{\n  \"schema\": 1, \"ar\": " << (o.ar ? "true" : "false")
            << ", \"audit\": " << (o.audit ? "true" : "false")
            << ", \"quiet\": " << (o.quiet ? "true" : "false")
            << ", \"resident_model\": " << (o.benchmark_plan.empty() ? "false" : "true")
            << ", \"logits_traced\": " << (o.trace_logits ? "true" : "false")
            << ", \"batch_probe_at\": " << o.probe_batch2_at
            << ", \"ssd\": " << (o.ssd ? "true" : "false")
            << ", \"direct_io\": " << (o.ssd && o.direct_io ? "true" : "false")
            << ", \"memory_budget_bytes\": " << o.memory_mib * 1024 * 1024
            << ", \"model_buffer_bytes\": " << memory.model_bytes << ", \"runtime_reserved_bytes\": " << o.reserve_mib * 1024 * 1024
            << ", \"runtime_accounted_bytes\": " << runtime_bytes
            << ", \"pruning\": " << (o.prune.enabled ? "true" : "false") << ", \"pruned_extensions\": " << pruned
            << ", \"draft_branches\": " << o.branches << ", \"tree_switches\": " << tree_switches
            << ", \"verification_batches\": " << verification_batches
            << ", \"approximate_expert_reuse\": " << (o.reuse_cap > 0 && (o.reuse_runtime || o.reuse_strength > 0) ? "true" : "false")
            << ",\n  \"ssd_expert_accesses\": " << io.expert_accesses << ", \"ssd_resident_hits\": " << io.resident_hits
            << ", \"ssd_read_requests\": " << io.read_requests << ", \"ssd_read_bytes\": " << io.read_bytes
            << ", \"ssd_fetch_ms\": " << io.fetch_us / 1000.0
            << ", \"ssd_draft_read_bytes\": " << io.draft_read_bytes << ", \"ssd_draft_fetch_ms\": " << io.draft_fetch_us / 1000.0
            << ", \"ssd_target_read_bytes\": " << io.target_read_bytes << ", \"ssd_target_fetch_ms\": " << io.target_fetch_us / 1000.0
            << ", \"ssd_prefill_read_bytes\": " << prefill_io.read_bytes
            << ", \"cuda_q8_ar_compat\": " << (o.gpu_layers > 0 ? "true" : "false")
            << ", \"model_load_ms\": " << model_load_ms
            << ",\n  \"generated\": " << emitted.size()
            << ", \"decode_tokens\": " << (emitted.empty() ? 0 : emitted.size() - 1)
            << ", \"accepted\": " << accepted_total << ", \"proposed\": " << proposed_total
            << ", \"rounds\": " << rounds << ", \"zero_rounds\": " << zero_rounds
            << ", \"partial_rounds\": " << partial_rounds << ", \"full_rounds\": " << full_rounds
            << ",\n  \"setup_ms\": " << setup_ms << ", \"prefill_ms\": " << prefill_ms
            << ", \"generation_ms\": " << generation_ms
            << ",\n  \"draft_ms\": " << draft_ms << ", \"verify_with_history_ms\": " << verify_ms
            << ", \"draft_branch_snapshot_ms\": " << fork_save_ms << ", \"draft_branch_restore_ms\": " << fork_restore_ms
            << ", \"draft_compute_ms\": " << draft_ms - fork_save_ms - fork_restore_ms
            << ", \"snapshot_ms\": " << snapshot_ms << ", \"restore_ms\": " << restore_ms
            << ", \"commit_ms\": " << commit_ms << ", \"ar_decode_ms\": " << ar_decode_ms
            << ", \"sampling_ms\": " << sampling_ms << "\n}\n";
        next_spec::require(out.good(), "cannot write timings output");
    }
    if (!o.tokens_file.empty()) {
        std::ofstream out(o.tokens_file);
        next_spec::require(out.good(), "cannot open token output");
        out << '[';
        for (size_t i = 0; i < emitted.size(); ++i) out << (i ? "," : "") << emitted[i];
        out << "]\n";
        next_spec::require(out.good(), "cannot write token output");
    }
    std::fprintf(stderr, "NEXT_SPEC_RESULT generated=%zu accepted=%d proposed=%d rounds=%d commit_replay=0\n",
                 emitted.size(), accepted_total, proposed_total, rounds);
    return 0;
}

static int run(const options & o) {
    const auto model_start = phase_timer::clock::now();
    next_spec::disable_graphs();
    next_spec::enable_q8_cuda_ar_compatibility();
    llama_backend_init();
    ggml_backend_load_all();
    if (o.ssd) {
        auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        next_spec::require(device && std::string(ggml_backend_dev_name(device)).find("CUDA") == 0,
            "SSD expert execution requires a CUDA GPU");
#ifdef _WIN32
        _putenv_s("S2MOE_SSD_DIRECT", o.direct_io ? "1" : "0");
#else
        setenv("S2MOE_SSD_DIRECT", o.direct_io ? "1" : "0", 1);
#endif
    }
    auto mp = llama_model_default_params();
    ggml_backend_dev_t cpu_only[] = {nullptr};
    if (o.gpu_layers == 0) mp.devices = cpu_only;
    mp.n_gpu_layers = o.gpu_layers;
    mp.use_ssd_moe = o.ssd;
    mp.hot_experts_path = o.hot.empty() ? nullptr : o.hot.c_str();
    if (o.ssd || o.no_mmap) mp.use_mmap = false;
    mp.ssd_weight_budget = o.memory_mib ? (o.memory_mib - o.reserve_mib) * 1024 * 1024 : 0;
    mp.moe_reuse_strength = o.reuse_strength;
    mp.moe_reuse_expert_cap = o.reuse_cap;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(o.model.c_str(), mp), llama_model_free);
    next_spec::require(model != nullptr, "model loading failed");
    char arch[64] = {};
    llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch));
#ifdef LLAMA_HYBRID_DRIVER
    next_spec::require(std::string(arch) == "qwen3next" || std::string(arch) == "qwen35moe", "expected a Qwen hybrid GGUF");
#elif defined(LLAMA_EXPERIMENTAL_QWEN35)
    next_spec::require(std::string(arch) == "qwen35moe", "this executable only supports qwen35moe GGUF models");
#else
    next_spec::require(std::string(arch) == "qwen3next", "this executable only supports qwen3next GGUF models");
#endif
    const double model_load_ms = std::chrono::duration<double, std::milli>(
        phase_timer::clock::now() - model_start).count();
    if (o.benchmark_plan.empty()) return generate(o, model.get(), model_load_ms);

    next_spec::require(o.quiet && !o.audit, "--benchmark-plan requires --quiet and auditing disabled");
    std::ifstream plan(o.benchmark_plan);
    next_spec::require(plan.good(), "cannot open benchmark plan");
    std::string line;
    size_t index = 0;
    while (std::getline(plan, line)) {
        if (line.empty()) continue;
        options item = o;
        std::string mode, extra, prompt;
        std::istringstream row(line);
        next_spec::require(bool(row >> mode >> item.draft >> item.experts
            >> std::quoted(item.tokens_file) >> std::quoted(item.timings_file)), "invalid benchmark row");
        if (row >> std::quoted(prompt)) item.prompt = prompt;
        next_spec::require(!(row >> extra) && (mode == "AR" || mode == "SPEC")
            && item.draft >= 0 && item.draft <= 31 && item.draft + 1 <= item.ubatch
            && item.experts > 0 && !item.tokens_file.empty() && !item.timings_file.empty(), "invalid benchmark case");
        item.ar = mode == "AR";
        std::fprintf(stderr, "NEXT_SPEC_CASE_BEGIN index=%zu mode=%s depth=%d experts=%d\n",
            index, mode.c_str(), item.draft, item.experts);
        const int result = generate(item, model.get(), model_load_ms);
        next_spec::require(result == 0, "benchmark case failed");
        std::fprintf(stderr, "NEXT_SPEC_CASE_END index=%zu\n", index++);
    }
    next_spec::require(index > 0 && !plan.bad(), "empty or unreadable benchmark plan");
    return 0;
}

#ifdef LLAMA_HYBRID_DRIVER
int llama_hybrid_main(int argc, char ** argv) {
#else
int main(int argc, char ** argv) {
#endif
    if (argc == 1 || (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h"))) {
        usage(argv[0]);
        return argc == 1 ? 1 : 0;
    }
    try { return run(parse(argc, argv)); }
    catch (const std::exception & error) {
#ifdef LLAMA_EXPERIMENTAL_QWEN35
        std::fprintf(stderr, "llama-speculative-qwen35: %s\n", error.what());
#else
        std::fprintf(stderr, "llama-speculative-next: %s\n", error.what());
#endif
        return 1;
    }
}
