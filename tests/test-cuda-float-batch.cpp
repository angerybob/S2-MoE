// Hybrid attention must not change reduction order with verification width:
// small float differences can cross subsequent Q8 activation-quantization bins.
#include "ggml.h"
#include "ggml-ar.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static std::vector<float> project(ggml_backend_t backend, ggml_tensor * weights,
        const std::vector<float> & input, int tokens, int heads) {
    context_ptr ctx(ggml_init({1 << 20, nullptr, true}), ggml_free);
    auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, weights->ne[0], tokens, heads);
    auto * y = ggml_mul_mat(ctx.get(), weights, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    ggml_mul_mat_set_ar_compat(y, true);
    auto * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, y);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "projection allocation failed");
    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "projection failed");
    std::vector<float> out(ggml_nelements(y));
    ggml_backend_tensor_get(y, out.data(), 0, ggml_nbytes(y));
    return out;
}

int main() try {
    ggml_backend_load_all();
    auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!device || std::string(ggml_backend_dev_name(device)).find("CUDA") != 0) return 77;
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device, nullptr), ggml_backend_free);
    require(bool(backend), "CUDA initialization failed");
    bool equal = true;
    for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16})
    for (int width : {64, 128, 256, 1024}) for (int heads : {1, 8}) for (int rows : {64, 1024}) {
        const int kv_heads = heads == 1 ? 1 : 2;
        context_ptr ctx(ggml_init({1 << 20, nullptr, true}), ggml_free);
        auto * weights = ggml_new_tensor_3d(ctx.get(), type, width, rows, kv_heads);
        buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        require(bool(buffer), "weight allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> raw(ggml_nelements(weights));
        for (size_t i = 0; i < raw.size(); ++i) raw[i] = 0.13f * std::sin(float(i) * 0.017f);
        std::vector<uint8_t> data(ggml_nbytes(weights));
        if (type == GGML_TYPE_F32) std::memcpy(data.data(), raw.data(), data.size());
        else if (type == GGML_TYPE_F16) ggml_fp32_to_fp16_row(raw.data(), reinterpret_cast<ggml_fp16_t *>(data.data()), raw.size());
        else ggml_fp32_to_bf16_row_ref(raw.data(), reinterpret_cast<ggml_bf16_t *>(data.data()), raw.size());
        ggml_backend_tensor_set(weights, data.data(), 0, data.size());
        for (int tokens : {2, 5, 9, 32}) {
            std::vector<float> input(width * tokens * heads), serial(rows * tokens * heads);
            for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(float(i) * 0.117f) * (0.5f + i % 7);
            const auto batch = project(backend.get(), weights, input, tokens, heads);
            for (int t = 0; t < tokens; ++t) {
                std::vector<float> one(width * heads);
                for (int h = 0; h < heads; ++h) std::copy_n(input.begin() + width * (t + tokens * h), width, one.begin() + width * h);
                const auto output = project(backend.get(), weights, one, 1, heads);
                for (int h = 0; h < heads; ++h) std::copy_n(output.begin() + rows * h, rows, serial.begin() + rows * (t + tokens * h));
            }
            float maximum = 0;
            for (size_t i = 0; i < batch.size(); ++i) {
                require(std::isfinite(batch[i]) && std::isfinite(serial[i]), "non-finite projection");
                maximum = std::max(maximum, std::abs(batch[i] - serial[i]));
            }
            std::fprintf(stderr, "FLOAT_AR type=%s width=%d rows=%d heads=%d tokens=%d max_abs=%.9g\n", ggml_type_name(type), width, rows, heads, tokens, maximum);
            equal = equal && maximum == 0;
        }
    }
    return equal ? 0 : 1;
} catch (const std::exception & error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
