from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))

import gguf  # noqa: E402


def test_dflash_architecture_contract():
    assert gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.DFLASH] == "dflash"

    tensors = gguf.MODEL_TENSORS[gguf.MODEL_ARCH.DFLASH]

    assert gguf.MODEL_TENSOR.FC in tensors
    assert gguf.MODEL_TENSOR.ENC_OUTPUT_NORM in tensors
    assert gguf.MODEL_TENSOR.OUTPUT_NORM in tensors
    assert gguf.MODEL_TENSOR.ATTN_Q_NORM in tensors
    assert gguf.MODEL_TENSOR.ATTN_K_NORM in tensors
