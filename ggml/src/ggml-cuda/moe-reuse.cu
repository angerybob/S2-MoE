#include "moe-reuse.cuh"

#include "ggml-cuda/common.cuh"
#include "ggml-impl.h"

static __global__ void moe_reuse_importance_kernel(
        const float * logits,
              float * out,
        int           n_experts,
        int           n_tokens,
        int           top_k,
        float         lambda,
        int           cap,
        int           runtime_strength) {
    const int tid      = threadIdx.x;
    const int threads  = blockDim.x;

    extern __shared__ unsigned char smem_raw[];
    float * g         = (float *) smem_raw;
    float * reduce    = g + threads;
    int   * selected  = (int   *)(reduce + threads);
    int   * idx_buf   = selected + threads;
    __shared__ float max_w_raw;
    __shared__ float effective_lambda;

    if (tid == 0) {
        max_w_raw = 0.0f;
        effective_lambda = lambda;
    }

    if (tid < n_experts) {
        g[tid] = 0.0f;
    }
    if (tid < threads) {
        selected[tid] = 0;
    }
    __syncthreads();

    if (runtime_strength && tid == 0 && top_k < n_experts) {
        float margin_sum = 0.0f;
        for (int b = 0; b < n_tokens; ++b) {
            const float * row = logits + b * n_experts;
            float top1 = -INFINITY;
            float top_k_plus_1 = -INFINITY;
            for (int rank = 0; rank <= top_k; ++rank) {
                float best = -INFINITY;
                int best_idx = -1;
                for (int e = 0; e < n_experts; ++e) {
                    if (selected[e]) continue;
                    if (row[e] > best || (row[e] == best && (best_idx < 0 || e < best_idx))) {
                        best = row[e]; best_idx = e;
                    }
                }
                selected[best_idx] = 1;
                if (rank == 0) top1 = best;
                top_k_plus_1 = best;
            }
            margin_sum += top1 - top_k_plus_1;
            for (int e = 0; e < n_experts; ++e) selected[e] = 0;
        }
        effective_lambda = margin_sum / (float) n_tokens;
    }
    __syncthreads();

    for (int b = 0; b < n_tokens; ++b) {
        const float v = tid < n_experts ? logits[b * n_experts + tid] : 0.0f;

        // sum
        reduce[tid] = tid < n_experts ? v : 0.0f;
        __syncthreads();
        for (int offset = threads / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                reduce[tid] += reduce[tid + offset];
            }
            __syncthreads();
        }
        const float sum = reduce[0];

        // max
        reduce[tid] = tid < n_experts ? v : -INFINITY;
        __syncthreads();
        for (int offset = threads / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const float other = reduce[tid + offset];
                const float cur   = reduce[tid];
                if (other > cur) {
                    reduce[tid] = other;
                }
            }
            __syncthreads();
        }
        const float mx = reduce[0];

        float w = 0.0f;
        if (tid == 0) {
            w = fmaxf(0.0f, mx - sum / (float) n_experts);
            max_w_raw = fmaxf(max_w_raw, w);
            reduce[0] = w;
        }
        __syncthreads();
        w = reduce[0];

        if (tid < n_experts && w > 0.0f) {
            g[tid] += w * v;
        }
        __syncthreads();
    }

    if (max_w_raw == 0.0f || effective_lambda <= 0.0f || cap <= 0 || n_tokens <= 1 || n_experts <= 1) {
        if (tid < n_experts) {
            for (int b = 0; b < n_tokens; ++b) {
                const int idx = b * n_experts + tid;
                out[idx] = logits[idx];
            }
        }
        return;
    }

    const float scale = 1.0f / max_w_raw;
    if (tid < n_experts) {
        g[tid] *= scale;
    }
    __syncthreads();

    int k_global = cap < n_experts ? cap : n_experts;
    if (k_global < top_k) {
        k_global = top_k;
    }

    for (int k = 0; k < k_global; ++k) {
        float val = (tid < n_experts && selected[tid] == 0) ? g[tid] : -INFINITY;
        reduce[tid] = val;
        idx_buf[tid] = tid;
        __syncthreads();

        for (int offset = threads / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const float other_val = reduce[tid + offset];
                const int   other_idx = idx_buf[tid + offset];
                const float cur_val   = reduce[tid];
                const int   cur_idx   = idx_buf[tid];
                if (other_val > cur_val || (other_val == cur_val && other_idx < cur_idx)) {
                    reduce[tid] = other_val;
                    idx_buf[tid] = other_idx;
                }
            }
            __syncthreads();
        }

        const int winner = idx_buf[0];
        if (tid == winner) {
            selected[tid] = 1;
        }
        __syncthreads();
    }

    if (tid < n_experts) {
        const float bias = selected[tid] ? effective_lambda : 0.0f;
        for (int b = 0; b < n_tokens; ++b) {
            const int idx = b * n_experts + tid;
            out[idx] = logits[idx] + bias;
        }
    }
}

void ggml_cuda_op_moe_reuse_two_pass(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t n_experts = src0->ne[0];
    const int64_t n_tokens  = src0->ne[1];

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == (size_t) n_experts * sizeof(float));
    GGML_ASSERT(dst->nb[0]  == sizeof(float));
    GGML_ASSERT(dst->nb[1]  == (size_t) n_experts * sizeof(float));

    const int top_k   = ggml_get_op_params_i32(dst, 0);
    const int cap     = ggml_get_op_params_i32(dst, 1);
    const float lambda = ggml_get_op_params_f32(dst, 2);
    const int runtime_strength = ggml_get_op_params_i32(dst, 3);

    GGML_ASSERT(top_k > 0 && top_k <= n_experts);

    const float * logits_d = (const float *) src0->data;
    float * out_d = (float *) dst->data;

    if ((!runtime_strength && lambda <= 0.0f) || cap <= 0 || n_tokens <= 1 || n_experts <= 1) {
        CUDA_CHECK(cudaMemcpyAsync(out_d, logits_d, sizeof(float) * n_experts * n_tokens,
                    cudaMemcpyDeviceToDevice, ctx.stream()));
        return;
    }

    int threads = 1;
    while (threads < n_experts) {
        threads <<= 1;
    }
    threads = MIN(threads, 1024);

    GGML_ASSERT(threads >= n_experts);

    const dim3 grid(1, 1, 1);
    const dim3 block(threads, 1, 1);
    const size_t shmem = threads * (2 * sizeof(float) + 2 * sizeof(int));

    moe_reuse_importance_kernel<<<grid, block, shmem, ctx.stream()>>>(
        logits_d, out_d, (int) n_experts, (int) n_tokens, top_k, lambda, cap, runtime_strength);
}
