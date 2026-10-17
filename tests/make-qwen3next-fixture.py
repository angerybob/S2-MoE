#!/usr/bin/env python3
"""Generate a small deterministic random-weight model; never downloads weights."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import gguf  # noqa: E402


def main():
    arch = sys.argv[3] if len(sys.argv) > 3 else "qwen3next"
    writer = gguf.GGUFWriter(sys.argv[1], arch)
    writer.add_name(f"{arch} random correctness fixture (not a language model)")
    writer.add_context_length(128)
    writer.add_embedding_length(32)
    writer.add_block_count(8)
    writer.add_feed_forward_length(32)
    writer.add_expert_feed_forward_length(32)
    writer.add_expert_shared_feed_forward_length(32)
    writer.add_expert_count(4)
    writer.add_expert_used_count(2)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_rope_dimension_count(8)
    if arch == "qwen35moe":
        writer.add_rope_dimension_sections([2, 1, 1, 0])
    writer.add_ssm_conv_kernel(4)
    writer.add_ssm_inner_size(32)
    writer.add_ssm_state_size(8)
    writer.add_ssm_time_step_rank(4)
    writer.add_ssm_group_count(2)
    # Byte-complete GPT-2 vocabulary, plus one merge, for CLI tokenization tests.
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = byte_values.copy()
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            chars.append(256 + extra)
            extra += 1
    tokens = ["<|endoftext|>"] + [chr(c) for c in chars] + ["ab"]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen35" if arch == "qwen35moe" else "gpt-2")
    writer.add_token_list(tokens)
    writer.add_token_types([3] + [1] * (len(tokens) - 1))
    writer.add_token_merges(["a b"])
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(0)
    writer.add_add_bos_token(False)
    rng = np.random.default_rng(20261017)
    weight_type = sys.argv[2] if len(sys.argv) > 2 else "f32"

    def tensor(name, shape, constant=None):
        value = rng.normal(0, 0.08, shape).astype(np.float32) if constant is None else np.full(shape, constant, np.float32)
        if len(shape) >= 2 and "ssm_conv1d" not in name and weight_type != "f32":
            dtype = {"f16": gguf.GGMLQuantizationType.F16, "bf16": gguf.GGMLQuantizationType.BF16}[weight_type]
            writer.add_tensor(name, gguf.quantize(value, dtype), raw_dtype=dtype)
        else:
            writer.add_tensor(name, value)

    tensor("token_embd.weight", (len(tokens), 32))
    tensor("output.weight", (len(tokens), 32))
    tensor("output_norm.weight", (32,), 1)
    for layer in range(8):
        p = f"blk.{layer}."
        tensor(p + "attn_norm.weight", (32,), 1)
        tensor(p + "post_attention_norm.weight", (32,), 1)
        if (layer + 1) % 4 == 0:
            tensor(p + "attn_q.weight", (64, 32))
            tensor(p + "attn_k.weight", (16, 32))
            tensor(p + "attn_v.weight", (16, 32))
            tensor(p + "attn_output.weight", (32, 32))
            tensor(p + "attn_q_norm.weight", (8,), 1)
            tensor(p + "attn_k_norm.weight", (8,), 1)
        else:
            tensor(p + "attn_qkv.weight", (64, 32))
            tensor(p + "attn_gate.weight", (32, 32))
            tensor(p + "ssm_conv1d.weight", (64, 4))
            tensor(p + "ssm_dt.bias", (4,), 0)
            tensor(p + "ssm_a", (4,), -1)
            if arch == "qwen35moe":
                tensor(p + "ssm_alpha.weight", (4, 32))
                tensor(p + "ssm_beta.weight", (4, 32))
            else:
                tensor(p + "ssm_ba.weight", (8, 32))
            tensor(p + "ssm_norm.weight", (8,), 1)
            tensor(p + "ssm_out.weight", (32, 32))
        tensor(p + "ffn_gate_inp.weight", (4, 32))
        for projection in ("gate", "up", "down"):
            tensor(p + f"ffn_{projection}_exps.weight", (4, 32, 32))
            tensor(p + f"ffn_{projection}_shexp.weight", (32, 32))
        tensor(p + "ffn_gate_inp_shexp.weight", (32,))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


if __name__ == "__main__":
    main()
