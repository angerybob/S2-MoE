#include "next-state.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

struct options {
    std::string model, prompt = "The capital of France is", hot, tokens_file, timings_file, benchmark_plan;
    int predict = 32, draft = 8, experts = 2, gpu_layers = 99, context = 512, ubatch = 32, threads = 8;
    int probe_batch2_at = -1;
    bool ssd = false, ar = false, audit = false, ignore_eos = false, quiet = false, trace_logits = false;
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

static void usage(const char * exe) {
    std::fprintf(stderr,
#ifdef LLAMA_EXPERIMENTAL_QWEN35
        "Experimental greedy Qwen3.5-MoE expert-only self-speculation\n"
#else
        "Experimental greedy Qwen3-Next expert-only self-speculation\n"
#endif
        "Usage: %s -m model.gguf [-p prompt] [-n 32] [-ngl 99] [-c 512] [-ub 32]\n"
#ifdef LLAMA_EXPERIMENTAL_QWEN35
        "       [--draft-max 8] [--draft-expert-topk 2]\n"
#else
        "       [--draft-max 8] [--draft-expert-topk 2] [--ssd-moe] [--hot-experts file]\n"
#endif
        "       [--audit-state-copies] [--ar] [--tokens-file file] [--ignore-eos] [-t 8]\n"
        "       [--timings-file file] [--quiet]\n"
        "       [--benchmark-plan file] (reuse model, fresh contexts for each case)\n"
        "       [--trace-logits] (diagnostic target top-two logits; affects timings)\n"
        "       [--probe-batch2-at output-index] (AR-only diagnostic; affects timings)\n"
        "One request/branch, shared hybrid memory, raw prompt, greedy sampling only.\n"
        "All recurrent/full-attention layers and shared experts remain active.\n"
#ifdef LLAMA_EXPERIMENTAL_QWEN35
        "CUDA graphs and graph reuse are disabled. Vision/MTP and SSD loading are unsupported.\n", exe);
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
        if (arg == "-m" || arg == "--model") o.model = value();
        else if (arg == "-p" || arg == "--prompt") o.prompt = value();
        else if (arg == "-n" || arg == "--predict") o.predict = integer();
        else if (arg == "-ngl" || arg == "--gpu-layers") o.gpu_layers = integer();
        else if (arg == "-c" || arg == "--ctx-size") o.context = integer();
        else if (arg == "-ub" || arg == "--ubatch-size") o.ubatch = integer();
        else if (arg == "-t" || arg == "--threads") o.threads = integer();
        else if (arg == "--draft-max") o.draft = integer();
        else if (arg == "--draft-expert-topk") o.experts = integer();
        else if (arg == "--hot-experts") o.hot = value();
        else if (arg == "--tokens-file") o.tokens_file = value();
        else if (arg == "--timings-file") o.timings_file = value();
        else if (arg == "--benchmark-plan") o.benchmark_plan = value();
        else if (arg == "--quiet") o.quiet = true;
        else if (arg == "--trace-logits") o.trace_logits = true;
        else if (arg == "--probe-batch2-at") o.probe_batch2_at = integer();
        else if (arg == "--ssd-moe") o.ssd = true;
        else if (arg == "--ar") o.ar = true;
        else if (arg == "--audit-state-copies") o.audit = true;
        else if (arg == "--ignore-eos") o.ignore_eos = true;
        else if (arg == "--draft-share-kv") { /* always shared in this entry point */ }
        else if (arg == "--temp") next_spec::require(value() == "0", "only --temp 0 is supported");
        else throw std::runtime_error("unsupported option: " + arg);
    }
    next_spec::require(!o.model.empty(), "-m is required");
    next_spec::require(o.predict >= 0 && o.context > 0 && o.threads > 0, "invalid generation limits");
    next_spec::require(o.draft >= 0 && o.draft <= 31, "--draft-max must be in [0, 31]");
    next_spec::require(o.ubatch >= o.draft + 1 && o.ubatch <= 512,
                       "-ub must fit root + draft in one ubatch and be at most 512");
    next_spec::require(o.experts > 0, "--draft-expert-topk must be positive");
    return o;
}

static int generate(const options & o, llama_model * model, double model_load_ms) {
    const bool profile = !o.timings_file.empty();
    const auto setup_start = phase_timer::clock::now();
    double prefill_ms = 0, draft_ms = 0, verify_ms = 0, snapshot_ms = 0;
    double restore_ms = 0, commit_ms = 0, ar_decode_ms = 0, sampling_ms = 0;
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
    }
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
        for (int off = 0; off < count; off += o.ubatch) {
            next_spec::decode(target.get(), {prompt.begin() + off, prompt.begin() + std::min(count, off + o.ubatch)}, off);
        }
        if (profile) llama_synchronize(target.get());
    }
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
        const int budget = std::min(o.draft, o.predict - int(emitted.size()));
        std::vector<llama_token> verification{root};
        llama_token input = root;
        {
            phase_timer timer(profile, draft_ms);
            for (int i = 0; i < budget; ++i) {
                next_spec::decode(draft.get(), {input}, pos + i);
                input = next_spec::greedy(draft.get());
                verification.push_back(input);
            }
        }
        proposed_total += budget;
        {
            phase_timer timer(profile, restore_ms);
            prefix->restore(target.get());
        }
        {
            phase_timer timer(profile, verify_ms);
            next_spec::require(llama_memory_history_begin(llama_get_memory(target.get()), 0, pos, verification.size()),
                               "could not begin verification history");
            next_spec::decode(target.get(), verification, pos, true);
            next_spec::require(llama_memory_history_finish(target.get()), "incomplete verification history");
        }
        int accepted = 0;
        {
            phase_timer timer(profile, sampling_ms);
            root = target_greedy(0);
            while (accepted < budget && verification[accepted + 1] == root) {
                if (!emit(root)) { stopped = true; break; }
                ++accepted;
                root = target_greedy(accepted);
            }
        }
        {
            phase_timer timer(profile, commit_ms);
            next_spec::require(llama_memory_history_commit(target.get(), 0, accepted + 1, o.audit),
                               "accepted-prefix history commit failed");
            if (profile) llama_synchronize(target.get());
        }
        if (accepted == 0) ++zero_rounds;
        else if (accepted == budget) ++full_rounds;
        else ++partial_rounds;
        pos += accepted + 1;
        accepted_total += accepted;
        ++rounds;
        if (stopped) break;
    }
    if (profile) llama_synchronize(target.get());
    const double generation_ms = std::chrono::duration<double, std::milli>(
        phase_timer::clock::now() - generation_start).count();
    if (!o.quiet) std::fputc('\n', stdout);
    if (profile) {
        std::ofstream out(o.timings_file);
        next_spec::require(out.good(), "cannot open timings output");
        out.precision(12);
        out << "{\n  \"schema\": 1, \"ar\": " << (o.ar ? "true" : "false")
            << ", \"audit\": " << (o.audit ? "true" : "false")
            << ", \"quiet\": " << (o.quiet ? "true" : "false")
            << ", \"resident_model\": " << (o.benchmark_plan.empty() ? "false" : "true")
            << ", \"logits_traced\": " << (o.trace_logits ? "true" : "false")
            << ", \"batch_probe_at\": " << o.probe_batch2_at
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
#ifdef LLAMA_EXPERIMENTAL_QWEN35
    next_spec::require(!o.ssd && o.hot.empty(), "Qwen3.5 SSD expert loading is not yet qualified");
#endif
    next_spec::disable_graphs();
    next_spec::enable_q8_cuda_ar_compatibility();
    llama_backend_init();
    ggml_backend_load_all();
    auto mp = llama_model_default_params();
    ggml_backend_dev_t cpu_only[] = {nullptr};
    if (o.gpu_layers == 0) mp.devices = cpu_only;
    mp.n_gpu_layers = o.gpu_layers;
    mp.use_ssd_moe = o.ssd;
    mp.hot_experts_path = o.hot.empty() ? nullptr : o.hot.c_str();
    mp.moe_reuse_strength = 0;
    mp.moe_reuse_expert_cap = 0;
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
        llama_model_load_from_file(o.model.c_str(), mp), llama_model_free);
    next_spec::require(model != nullptr, "model loading failed");
    char arch[64] = {};
    llama_model_meta_val_str(model.get(), "general.architecture", arch, sizeof(arch));
#ifdef LLAMA_EXPERIMENTAL_QWEN35
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

int main(int argc, char ** argv) {
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
