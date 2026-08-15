# Native draft-model decoders

`llama-speculative` contains the native EAGLE3 encoder/decoder path. The target context extracts the layers requested by the draft GGUF metadata, the encoder fuses those features, and the decoder uses confidence-controlled early stopping up to `--draft`.

```bash
cmake --build build --target llama-speculative -j

build/bin/llama-speculative \
  -m /path/to/target.gguf \
  -md /path/to/eagle3-draft.gguf \
  --eagle3 -ngl 99 -ngld 99 \
  --draft 8 --draft-p-min 0.5 \
  -n 32 -p "Explain speculative decoding."
```

For split-expert target GGUFs, add `--ssd-moe` and the same hot-expert/offload options used by the target baseline. Cascade-style draft-width selection can additionally be enabled with `--moe-utility-spec`.

The same executable provides S²-MoE, ExpertSkip, LayerSkip, Cascade-style utility-driven speculation, EAGLE3, DFlash, and Domino.

## DFlash and Domino

DFlash and Domino use the same native draft-model interface. Pass a compatible
draft GGUF with `--spec-type draft-dflash`; Domino metadata is detected by the
loader and automatically selects the Domino GPU sampling path.

```bash
build/bin/llama-speculative \
  -m /path/to/target.gguf \
  -md /path/to/dflash-or-domino-draft.gguf \
  --spec-type draft-dflash -ngl 99 -ngld 99 \
  --draft 8 -n 64 -p "Explain speculative decoding."
```

Dataset evaluation uses the same executable and accepts `--dataset` together
with `--n-questions`.
