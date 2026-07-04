#include "llama-model-dflash.h"

#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-model.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

void dflash_validate_config(
        uint32_t block_size,
        uint32_t mask_token_id,
        uint32_t vocab_size,
        uint32_t target_layer_count,
        const std::vector<int> & target_layers) {
    if (block_size < 2) {
        throw std::invalid_argument("DFlash block_size must be at least 2");
    }
    if (mask_token_id >= vocab_size) {
        throw std::invalid_argument("DFlash mask_token_id is outside the input vocabulary");
    }
    if (target_layers.empty()) {
        throw std::invalid_argument("DFlash target_layers must not be empty");
    }
    for (int layer : target_layers) {
        if (layer <= 0 || static_cast<uint32_t>(layer) > target_layer_count) {
            throw std::invalid_argument("DFlash target layer is outside the target model");
        }
    }
}

void dflash_append_target_features(
        std::vector<float> & destination,
        const std::vector<const float *> & layers,
        size_t n_tokens,
        size_t n_embd) {
    const size_t n_layers = layers.size();
    const size_t old_size = destination.size();
    destination.resize(old_size + n_tokens * n_layers * n_embd);

    for (size_t token = 0; token < n_tokens; ++token) {
        for (size_t layer = 0; layer < n_layers; ++layer) {
            const float * src = layers[layer] + token * n_embd;
            float * dst = destination.data() + old_size + (token * n_layers + layer) * n_embd;
            std::memcpy(dst, src, n_embd * sizeof(float));
        }
    }
}

void dflash_configure_swa(
        llama_hparams & hparams,
        uint32_t sliding_window,
        const std::vector<bool> & pattern) {
    if (sliding_window == 0) {
        return;
    }
    if (pattern.size() != hparams.n_layer) {
        throw std::invalid_argument("DFlash sliding-window pattern must match the decoder layer count");
    }

    hparams.n_swa = sliding_window;
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.rope_freq_base_train_swa = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
    for (uint32_t il = 0; il < hparams.n_layer; ++il) {
        hparams.swa_layers[il] = pattern[il];
    }
}

void dflash_expand_logits(
        std::vector<float> & target_logits,
        const std::vector<float> & draft_logits,
        const std::vector<int32_t> & draft_to_target_delta,
        size_t target_vocab_size) {
    const size_t draft_vocab_size = draft_to_target_delta.size();
    if (draft_vocab_size == 0 || draft_logits.size() % draft_vocab_size != 0) {
        throw std::invalid_argument("DFlash logits do not match the draft vocabulary");
    }

    const size_t n_rows = draft_logits.size() / draft_vocab_size;
    target_logits.assign(
        n_rows * target_vocab_size,
        -std::numeric_limits<float>::infinity());
    for (size_t row = 0; row < n_rows; ++row) {
        for (size_t draft_id = 0; draft_id < draft_vocab_size; ++draft_id) {
            const int64_t target_id =
                static_cast<int64_t>(draft_id) + draft_to_target_delta[draft_id];
            if (target_id < 0 || static_cast<size_t>(target_id) >= target_vocab_size) {
                throw std::invalid_argument("DFlash draft-to-target token mapping is out of range");
            }
            target_logits[row * target_vocab_size + target_id] =
                draft_logits[row * draft_vocab_size + draft_id];
        }
    }
}

llm_build_dflash_encode::llm_build_dflash_encode(
        const llama_model & model,
        const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_features = model.dflash_target_layers.size() * n_embd;

    auto inp = std::make_unique<llm_graph_input_embd>();
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_features, n_tokens);
    ggml_set_input(inp->embd);

    ggml_tensor * cur = inp->embd;
    cb(cur, "inp_embd", -1);
    res->add_input(std::move(inp));

    cur = build_lora_mm(model.fc, cur);
    cb(cur, "fc_out", -1);
    cur = build_norm(cur, model.output_norm_enc, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "enc_norm_out", -1);

    ggml_set_output(cur);
    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}

llm_build_dflash_decode::llm_build_dflash_decode(
        const llama_model & model,
        const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);

    ggml_tensor * inp_pos = build_inp_pos();
    const bool use_iswa = hparams.swa_type != LLAMA_SWA_TYPE_NONE;
    llm_graph_input_attn_kv * inp_attn = nullptr;
    llm_graph_input_attn_kv_iswa * inp_attn_iswa = nullptr;
    if (use_iswa) {
        inp_attn_iswa = build_attn_inp_kv_iswa();
    } else {
        inp_attn = build_attn_inp_kv();
    }
    const float kq_scale = 1.0f / std::sqrt(float(n_embd_head));

    if (ubatch.embd) {
        auto inp = std::make_unique<llm_graph_input_embd>();
        inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_input(inp->embd);
        ggml_tensor * inp_g = inp->embd;
        cb(inp_g, "inp_g_embeddings", -1);
        res->add_input(std::move(inp));

        for (int il = 0; il < n_layer; ++il) {
            const auto & layer = model.layers[il];
            ggml_tensor * k_cur = build_lora_mm(layer.wk, inp_g);
            ggml_tensor * v_cur = build_lora_mm(layer.wv, inp_g);
            if (layer.bk) {
                k_cur = ggml_add(ctx0, k_cur, layer.bk);
            }
            if (layer.bv) {
                v_cur = ggml_add(ctx0, v_cur, layer.bv);
            }
            k_cur = ggml_reshape_3d(ctx0, k_cur, n_embd_head, n_head_kv, n_tokens);
            v_cur = ggml_reshape_3d(ctx0, v_cur, n_embd_head, n_head_kv, n_tokens);
            k_cur = build_norm(k_cur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
            k_cur = ggml_rope_ext(
                    ctx0, k_cur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            if (use_iswa) {
                const bool is_swa = hparams.is_swa(il);
                const auto * kv = is_swa ? inp_attn_iswa->mctx->get_swa() : inp_attn_iswa->mctx->get_base();
                ggml_tensor * k_idxs = is_swa ? inp_attn_iswa->get_k_idxs_swa() : inp_attn_iswa->get_k_idxs();
                ggml_tensor * v_idxs = is_swa ? inp_attn_iswa->get_v_idxs_swa() : inp_attn_iswa->get_v_idxs();
                ggml_build_forward_expand(gf, kv->cpy_k(ctx0, k_cur, k_idxs, il));
                ggml_build_forward_expand(gf, kv->cpy_v(ctx0, v_cur, v_idxs, il));
            } else {
                ggml_build_forward_expand(
                        gf, inp_attn->mctx->cpy_k(ctx0, k_cur, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(
                        gf, inp_attn->mctx->cpy_v(ctx0, v_cur, inp_attn->get_v_idxs(), il));
            }
        }

        res->t_embd = inp_g;
        ggml_build_forward_expand(gf, inp_g);
        return;
    }

    ggml_tensor * token_embd = model.tok_embd ? model.tok_embd : model.target_tok_embd;
    GGML_ASSERT(token_embd != nullptr && "DFlash requires token embeddings");

    auto inp = std::make_unique<llm_graph_input_embd>();
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    ggml_tensor * inp_l = ggml_get_rows(ctx0, token_embd, inp->tokens);
    cb(inp_l, "inp_noise_embd", -1);
    res->add_input(std::move(inp));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * cur = build_norm(inp_l, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        ggml_tensor * q_cur = build_lora_mm(layer.wq, cur);
        ggml_tensor * k_cur = build_lora_mm(layer.wk, cur);
        ggml_tensor * v_cur = build_lora_mm(layer.wv, cur);
        if (layer.bq) {
            q_cur = ggml_add(ctx0, q_cur, layer.bq);
        }
        if (layer.bk) {
            k_cur = ggml_add(ctx0, k_cur, layer.bk);
        }
        if (layer.bv) {
            v_cur = ggml_add(ctx0, v_cur, layer.bv);
        }
        q_cur = ggml_reshape_3d(ctx0, q_cur, n_embd_head, n_head, n_tokens);
        k_cur = ggml_reshape_3d(ctx0, k_cur, n_embd_head, n_head_kv, n_tokens);
        v_cur = ggml_reshape_3d(ctx0, v_cur, n_embd_head, n_head_kv, n_tokens);
        q_cur = build_norm(q_cur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        k_cur = build_norm(k_cur, layer.attn_k_norm, nullptr, LLM_NORM_RMS, il);
        q_cur = ggml_rope_ext(
                ctx0, q_cur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        k_cur = ggml_rope_ext(
                ctx0, k_cur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

        cur = use_iswa
            ? build_attn(
                    inp_attn_iswa, layer.wo, layer.bo,
                    q_cur, k_cur, v_cur,
                    nullptr, nullptr, nullptr, kq_scale, il)
            : build_attn(
                    inp_attn, layer.wo, layer.bo,
                    q_cur, k_cur, v_cur,
                    nullptr, nullptr, nullptr, kq_scale, il);
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inp_l);
        cur = build_norm(ffn_inp, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_ffn(
                cur,
                layer.ffn_up, nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        inp_l = ggml_add(ctx0, cur, ffn_inp);
        cb(inp_l, "layer_out", il);
    }

    ggml_tensor * cur = build_norm(inp_l, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    ggml_tensor * output = model.output ? model.output : model.target_output;
    GGML_ASSERT(output != nullptr && "DFlash requires an output projection");
    cur = build_lora_mm(output, cur);
    cb(cur, "result_output", -1);
    res->t_embd = inp_l;
    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
}
