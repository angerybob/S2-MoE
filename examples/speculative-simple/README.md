# Native EAGLE3 baseline

`llama-speculative-eagle3` contains the native EAGLE3 encoder/decoder path. The target context extracts the three layers requested by the draft GGUF metadata, the encoder fuses those features, and the decoder uses confidence-controlled early stopping up to `--draft`.

```bash
cmake --build build --target llama-speculative-eagle3 -j

build/bin/llama-speculative-eagle3 \
  -m /path/to/target.gguf \
  -md /path/to/eagle3-draft.gguf \
  --eagle3 -ngl 99 -ngld 99 \
  --draft 8 --draft-p-min 0.5 \
  -n 32 -p "Explain speculative decoding."
```

For split-expert target GGUFs, add `--ssd-moe` and the same hot-expert/offload options used by the target baseline. Cascade-style draft-width selection can additionally be enabled with `--moe-utility-spec`.

Use `examples/speculative/` and `build/bin/llama-speculative` for S2-MoE, ExpertSkip, LayerSkip, and Cascade-style utility-driven speculation.
