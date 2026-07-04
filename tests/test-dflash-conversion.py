from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402
import convert_hf_to_gguf as converter  # noqa: E402


def test_dflash_architecture_contract():
    assert gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.DFLASH] == "dflash"

    tensors = gguf.MODEL_TENSORS[gguf.MODEL_ARCH.DFLASH]

    assert gguf.MODEL_TENSOR.FC in tensors
    assert gguf.MODEL_TENSOR.ENC_OUTPUT_NORM in tensors
    assert gguf.MODEL_TENSOR.OUTPUT_NORM in tensors
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
