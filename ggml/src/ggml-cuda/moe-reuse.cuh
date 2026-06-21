#pragma once

#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_moe_reuse_two_pass(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
