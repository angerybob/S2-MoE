// Quantized hybrid verification must use the same projection arithmetic
// as serial AR when the graph explicitly requests GGML_PREC_F32.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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
        int channels, int tokens, int used, const float * input, const int32_t * experts, bool strided) {
    context_ptr ctx(ggml_init({1 << 20, nullptr, true}), ggml_free);
    require(bool(ctx), "graph context allocation failed");
    auto * x = channels == 0 ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, weights->ne[0], tokens)
        : ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, weights->ne[0], channels, tokens);
    auto * ids_storage = channels == 0 ? nullptr : ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32,
        strided ? weights->ne[2] : used, tokens);
    auto * ids = !ids_storage ? nullptr : ggml_view_2d(ctx.get(), ids_storage, used, tokens, ids_storage->nb[1], 0);
    ggml_tensor dense_weights = *weights;
    dense_weights.ne[2] = 1;
    auto * y = channels == 0 ? ggml_mul_mat(ctx.get(), &dense_weights, x) : ggml_mul_mat_id(ctx.get(), weights, x, ids);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    auto * graph = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(graph, y);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "graph buffer allocation failed");
    ggml_backend_tensor_set(x, input, 0, ggml_nbytes(x));
    if (ids_storage) {
        std::vector<int32_t> padded(ggml_nelements(ids_storage), -1);
        for (int t = 0; t < tokens; ++t) {
            std::copy(experts + t * used, experts + (t + 1) * used, padded.begin() + t * ids_storage->ne[0]);
        }
        ggml_backend_tensor_set(ids_storage, padded.data(), 0, ggml_nbytes(ids_storage));
    }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "projection failed");
    std::vector<float> result(ggml_nelements(y));
    ggml_backend_tensor_get(y, result.data(), 0, ggml_nbytes(y));
    return result;
}

int main() {
    try {
#ifdef _WIN32
        _putenv_s("GGML_CUDA_Q8_AR_COMPAT", "1");
#else
        setenv("GGML_CUDA_Q8_AR_COMPAT", "1", 1);
#endif
        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!device || std::string(ggml_backend_dev_name(device)).find("CUDA") != 0) {
            std::fprintf(stderr, "SKIP: CUDA device unavailable\n");
            return 77;
        }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            ggml_backend_dev_init(device, nullptr), ggml_backend_free);
        require(bool(backend), "CUDA initialization failed");
        constexpr int width = 128, rows = 64, count = 16, used = 8;
        context_ptr ctx(ggml_init({4 * ggml_tensor_overhead(), nullptr, true}), ggml_free);
        require(bool(ctx), "weight context allocation failed");
        auto * weights = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q8_0, width, rows, count);
        buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        require(bool(buffer), "weight buffer allocation failed");
        ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> raw(width * rows * count);
        for (size_t i = 0; i < raw.size(); ++i) raw[i] = 0.13f * std::sin(float(i) * 0.017f);
        std::vector<uint8_t> quantized(ggml_nbytes(weights));
        require(ggml_quantize_chunk(GGML_TYPE_Q8_0, raw.data(), quantized.data(), 0,
            rows * count, width, nullptr) == quantized.size(), "weight quantization failed");
        ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
        bool all_equal = true;
        for (int channels : {0, 1, used}) {
            for (int tokens : {2, 5, 9, 32}) {
                for (int strided = 0; strided < (channels == 0 ? 1 : 2); ++strided) {
                    const int input_channels = std::max(1, channels);
                    std::vector<float> input(width * input_channels * tokens);
                    std::vector<int32_t> experts(used * tokens);
                    for (size_t i = 0; i < input.size(); ++i) {
                        input[i] = std::sin(float(i) * 0.117f) * (0.5f + i % 7);
                    }
                    for (int t = 0; t < tokens; ++t) {
                        for (int k = 0; k < used; ++k) experts[t * used + k] = (t * 3 + k * 7) % count;
                    }
                    const auto batch = project(backend.get(), weights, channels, tokens, used,
                        input.data(), experts.data(), strided != 0);
                    std::vector<float> serial;
                    for (int t = 0; t < tokens; ++t) {
                        const auto one = project(backend.get(), weights, channels, 1, used,
                            input.data() + t * width * input_channels, experts.data() + t * used, strided != 0);
                        serial.insert(serial.end(), one.begin(), one.end());
                    }
                    float max_abs = 0;
                    for (size_t i = 0; i < batch.size(); ++i) {
                        require(std::isfinite(batch[i]) && std::isfinite(serial[i]), "non-finite projection");
                        max_abs = std::max(max_abs, std::abs(batch[i] - serial[i]));
                    }
                    std::fprintf(stderr, "Q8 CUDA channels=%d tokens=%d strided_ids=%d max_abs=%.9g\n",
                        channels, tokens, strided, max_abs);
                    all_equal = all_equal && max_abs == 0;
                }
            }
        }
        return all_equal ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
