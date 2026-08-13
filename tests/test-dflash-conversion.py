from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402
import convert_hf_to_gguf as converter  # noqa: E402


def test_dflash_architecture_contract():
    assert gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.DFLASH] == "dflash"

    tensors = gguf.MODEL_TENSORS[gguf.MODEL_ARCH.DFLASH]

    assert gguf.MODEL_TENSOR.FC in tensors
    assert gguf.MODEL_TENSOR.ENC_OUTPUT_NORM in tensors
    assert gguf.MODEL_TENSOR.OUTPUT_NORM in tensors
    assert gguf.MODEL_TENSOR.TOKEN_EMBD in tensors
    assert gguf.MODEL_TENSOR.OUTPUT in tensors
    assert gguf.MODEL_TENSOR.ATTN_Q_NORM in tensors
    assert gguf.MODEL_TENSOR.ATTN_K_NORM in tensors


def test_normalize_zlab_dflash_config():
    config = {
        "architectures": ["DFlashDraftModel"],
        "block_size": 10,
        "dflash_config": {
            "mask_token_id": 200000,
            "target_layer_ids": [1, 9, 17, 25, 33],
        },
        "hidden_size": 2880,
        "num_hidden_layers": 8,
        "vocab_size": 201088,
    }

    normalized = converter.normalize_dflash_config(config)

    assert normalized.block_size == 10
    assert normalized.mask_token_id == 200000
    assert normalized.target_layers == [2, 10, 18, 26, 34]
    assert normalized.draft_vocab_size == 201088
    assert normalized.decoder_config["hidden_size"] == 2880
    assert normalized.decoder_config["architectures"] == ["Qwen3ForCausalLM"]


def test_normalize_zlab_dflash_tensor_names():
    expected = {
        "fc.weight": "fc.weight",
        "hidden_norm.weight": "enc.output_norm.weight",
        "model.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight",
        "model.layers.0.self_attn.q_norm.weight": "blk.0.attn_q_norm.weight",
        "model.layers.0.mlp.down_proj.weight": "blk.0.ffn_down.weight",
        "model.norm.weight": "output_norm.weight",
    }

    actual = {
        name: converter.normalize_dflash_tensor_name(name)
        for name in expected
    }

    assert actual == expected


def test_normalize_speculators_dflash_config():
    config = {
        "architectures": ["DFlashDraftModel"],
        "aux_hidden_state_layer_ids": [1, 12, 23, 34, 45],
        "block_size": 8,
        "draft_vocab_size": 32000,
        "mask_token_id": 151669,
        "speculators_config": {
            "verifier": {"name_or_path": "Qwen/Qwen3-30B-A3B"},
        },
        "transformer_layer_config": {
            "hidden_size": 2048,
            "layer_types": ["sliding_attention"] * 5,
            "num_hidden_layers": 5,
            "rope_parameters": {
                "rope_theta": 1_000_000,
                "rope_type": "default",
            },
            "sliding_window": 2048,
            "vocab_size": 151936,
        },
    }

    normalized = converter.normalize_dflash_config(config)

    assert normalized.block_size == 8
    assert normalized.mask_token_id == 151669
    # Speculators stores the verifier extraction indices directly.  Unlike the
    # older Z-Lab target_layer_ids, these have already had the hidden-state
    # tuple offset applied.
    assert normalized.target_layers == [1, 12, 23, 34, 45]
    assert normalized.draft_vocab_size == 32000
    assert normalized.target_model == "Qwen/Qwen3-30B-A3B"
    assert normalized.decoder_config["hidden_size"] == 2048
    assert normalized.decoder_config["rope_theta"] == 1_000_000
    assert normalized.decoder_config["sliding_window"] == 2048
    assert normalized.decoder_config["architectures"] == ["Qwen3ForCausalLM"]
    assert converter.dflash_sliding_window_metadata(normalized.decoder_config) == (
        2048,
        [True, True, True, True, True],
    )


def test_normalize_speculators_dflash_tensor_names():
    expected = {
        "d2t": "d2t",
        "t2d": None,
        "embed_tokens.weight": "token_embd.weight",
        "lm_head.weight": "output.weight",
        "norm.weight": "output_norm.weight",
        "layers.4.self_attn.o_proj.weight": "blk.4.attn_output.weight",
    }

    actual = {
        name: converter.normalize_dflash_tensor_name(name)
        for name in expected
    }

    assert actual == expected


def test_normalize_domino_dflash_contract():
    config = {
        "architectures": ["DFlashDraftModel"],
        "block_size": 8,
        "dflash_config": {
            "mask_token_id": 151669,
            "target_layer_ids": [1, 9, 17, 25, 33],
            "projector_type": "domino",
            "pure_draft_prefix_len": 1,
            "shift_label": True,
        },
        "hidden_size": 2048,
        "num_hidden_layers": 5,
        "vocab_size": 151936,
    }

    normalized = converter.normalize_dflash_config(config)

    assert normalized.projector_type == "domino"
    assert normalized.pure_draft_prefix_len == 1
    assert normalized.shift_label is True
    assert converter.normalize_dflash_tensor_name("prefix_gru.weight_ih_l0") == "domino.prefix_gru.weight_ih_l0"
    assert converter.normalize_dflash_tensor_name("prefix_gru.weight_hh_l0") == "domino.prefix_gru.weight_hh_l0"
    assert converter.normalize_dflash_tensor_name("embed_proj.0.weight") == "domino.embed_proj.0.weight"
    assert converter.normalize_dflash_tensor_name("embed_proj.2.weight") == "domino.embed_proj.2.weight"
