// Compile this unchanged against the candidate and an independent upstream
// llama.cpp checkout to compare text tokenization and ordinary greedy decoding.
#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static bool trace_tensor(ggml_tensor * tensor, bool ask, void * data) {
    if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor) || ggml_nelements(tensor) > 100000) return false;
    if (ask) return true;
    auto & trace = *static_cast<std::map<std::string, std::vector<float>> *>(data);
    auto & values = trace[tensor->name];
    values.resize(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::fprintf(stderr, "usage: %s model prompt output.json tokens ubatch\n", argv[0]);
        return 1;
    }
    try {
        llama_backend_init();
        ggml_backend_load_all();
        const int predict = std::stoi(argv[4]);
        auto mp = llama_model_default_params();
        mp.vocab_only = predict == 0;
        ggml_backend_dev_t cpu_only[] = {nullptr};
        mp.devices = cpu_only;
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model loading failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        const std::string prompt = argv[2];
        std::vector<llama_token> input(prompt.size() + 16);
        int count = llama_tokenize(vocab, prompt.data(), prompt.size(), input.data(), input.size(), true, true);
        if (count < 0) {
            input.resize(-count);
            count = llama_tokenize(vocab, prompt.data(), prompt.size(), input.data(), input.size(), true, true);
        }
        require(count > 0, "tokenization failed");
        input.resize(count);
        require(predict >= 0 && count + predict <= 256, "invalid generation length");
        if (predict == 0) {
            std::ofstream out(argv[3]);
            out << "{\"prompt_tokens\":[";
            for (size_t i = 0; i < input.size(); ++i) out << (i ? "," : "") << input[i];
            out << "],\"generated_tokens\":[]}\n";
            require(out.good(), "cannot write vocabulary-only output");
            return 0;
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = 256;
        cp.n_batch = 256;
        cp.n_ubatch = std::stoi(argv[5]);
        cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 4;
        cp.op_offload = false;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::map<std::string, std::vector<float>> trace;
        if (std::getenv("QWEN35_TRACE")) {
            cp.cb_eval = trace_tensor;
            cp.cb_eval_user_data = &trace;
        }
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "context creation failed");
        auto decode = [&](const std::vector<llama_token> & tokens, int first) {
            auto batch = llama_batch_init(tokens.size(), 0, 1);
            batch.n_tokens = tokens.size();
            // Upstream Qwen3.5 uses four-axis IMRoPE; all axes agree for text.
            // llama_batch_init allocates one axis, so replace it explicitly.
            const int axes = (int(llama_model_rope_type(model.get())) & 8) ? 4 : 1;
            std::free(batch.pos);
            batch.pos = static_cast<llama_pos *>(std::malloc(sizeof(llama_pos) * tokens.size() * axes));
            require(batch.pos != nullptr, "cannot allocate text positions");
            for (int i = 0; i < batch.n_tokens; ++i) {
                batch.token[i] = tokens[i];
                for (int axis = 0; axis < axes; ++axis) batch.pos[axis * tokens.size() + i] = first + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = i + 1 == batch.n_tokens;
            }
            const int rc = llama_decode(ctx.get(), batch);
            llama_batch_free(batch);
            require(rc == 0, "ordinary AR decode failed");
        };
        for (int first = 0; first < count; first += cp.n_ubatch) {
            decode({input.begin() + first, input.begin() + std::min<int>(count, first + cp.n_ubatch)}, first);
        }
        std::vector<llama_token> output, forced;
        if (const char * path = std::getenv("QWEN35_FORCE_TOKENS")) {
            std::ifstream in(path);
            require(in.good(), "cannot open forced-token reference");
            for (llama_token token; in >> token;) forced.push_back(token);
            require(int(forced.size()) >= predict, "forced-token reference is too short");
        }
        std::vector<std::vector<float>> saved_logits;
        const bool dump_logits = std::getenv("QWEN35_DUMP_LOGITS") != nullptr;
        require(!dump_logits || llama_vocab_n_tokens(vocab) <= 1024, "logit dumping is fixture-only");
        for (int i = 0; i < predict; ++i) {
            const float * logits = llama_get_logits_ith(ctx.get(), -1);
            require(logits != nullptr, "missing AR logits");
            const int n_vocab = llama_vocab_n_tokens(vocab);
            for (int k = 0; k < n_vocab; ++k) require(std::isfinite(logits[k]), "non-finite AR logits");
            const llama_token token = std::max_element(logits, logits + n_vocab) - logits;
            output.push_back(token);
            if (dump_logits) saved_logits.emplace_back(logits, logits + n_vocab);
            if (i + 1 < predict) decode({forced.empty() ? token : forced[i]}, count + i);
        }
        std::ofstream out(argv[3]);
        require(out.good(), "cannot open reference output");
        auto array = [&](const std::vector<llama_token> & tokens) {
            out << '[';
            for (size_t i = 0; i < tokens.size(); ++i) out << (i ? "," : "") << tokens[i];
            out << ']';
        };
        out << "{\"prompt_tokens\":";
        array(input);
        out << ",\"generated_tokens\":";
        array(output);
        if (dump_logits) {
            out << ",\"step_logits\":[" << std::setprecision(9);
            for (size_t i = 0; i < saved_logits.size(); ++i) {
                out << (i ? ",[" : "[");
                for (size_t j = 0; j < saved_logits[i].size(); ++j) out << (j ? "," : "") << saved_logits[i][j];
                out << ']';
            }
            out << ']';
        }
        out << "}\n";
        require(out.good(), "cannot write reference output");
        if (const char * filename = std::getenv("QWEN35_TRACE")) {
            std::ofstream dump(filename);
            dump << '{' << std::setprecision(9);
            bool first = true;
            for (const auto & item : trace) {
                dump << (first ? "" : ",") << '\"' << item.first << "\":[";
                first = false;
                for (size_t i = 0; i < item.second.size(); ++i) dump << (i ? "," : "") << item.second[i];
                dump << ']';
            }
            dump << "}\n";
            require(dump.good(), "cannot write trace output");
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
