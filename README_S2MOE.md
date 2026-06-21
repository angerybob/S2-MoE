# S2-MoE notes

## Build

CPU-only build:

```bash
cmake -S . -B build -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-speculative llama-speculative-eagle3 -j
```

CUDA build:

```bash
cmake -S . -B build-cuda -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda --target llama-speculative llama-speculative-eagle3 -j
```

## Main executable

`build/bin/llama-speculative` contains:

- Auto: use the target model without drafting.
- ExpertSkip (ES): use the sparse self-drafter without adaptive expansion or reuse-aware gating.
- LayerSkip (LS): use configured draft layer skipping.
- Cascade: enable utility-driven draft-width selection with `--moe-utility-spec`.
- S2-MoE: enable KV sharing, adaptive expansion/pruning, and reuse-aware gating.

The JSON runner is:

```bash
python3 examples/speculative/run_spec.py --config examples/speculative/spec_config_olmoe2.json
```

The config files under `examples/speculative/` are templates.  Replace `/path/to/*.gguf` and `/path/to/hot_experts.json` with local model and hot-expert files.

## EAGLE3 baseline

`build/bin/llama-speculative-eagle3` provides the fully integrated native EAGLE3 path, including GGUF `eagle3` architecture loading, target-layer feature extraction, feature-fusion encoding, decoder guidance, draft-to-target vocabulary mapping, and confidence-controlled early stopping:

```bash
build/bin/llama-speculative-eagle3 /path/to/target.gguf /path/to/eagle3_draft.gguf
```

Cascade-style dynamic width is available through `--moe-utility-spec`. The EAGLE3 target remains a separate executable because it uses a dedicated encoder/decoder draft model, while S2-MoE uses the sparse self-draft path.

