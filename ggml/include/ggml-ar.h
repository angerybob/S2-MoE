#pragma once

#include "ggml.h"

// Optional dispatch hint in a previously unused MUL_MAT / MUL_MAT_ID parameter.
// Compatible CUDA F32 operations retain serial-vector reduction geometry over
// token columns, including grouped heads. Other backends may ignore the hint;
// it neither changes requested precision nor promises cross-backend identity.
enum { GGML_AR_COMPAT_OP_PARAM = 1 };

static inline void ggml_mul_mat_set_ar_compat(struct ggml_tensor * tensor, bool enabled) {
    GGML_ASSERT(tensor->op == GGML_OP_MUL_MAT || tensor->op == GGML_OP_MUL_MAT_ID);
    tensor->op_params[GGML_AR_COMPAT_OP_PARAM] = enabled ? 1 : 0;
}
