# Baseline and decoder commands

These commands use one prompt for brevity. Replace `-p "$PROMPT"` with `--dataset "$DATASET" --n-questions "$N"` for JSONL evaluation. Add `--ssd-moe --hot-experts "$HOT"` whenever the target uses split experts and SSD offloading.

```bash
TARGET=/path/to/target.gguf
SELF_DRAFT="$TARGET"
PROMPT="Explain speculative decoding."
COMMON=(-m "$TARGET" -p "$PROMPT" -n 64 -c 2048 -ngl 99 -t 8 --no-mmap)
```

## Auto

Target-only autoregressive decoding:

```bash
./build/bin/llama-cli "${COMMON[@]}" --single-turn
```

## ExpertSkip

Use fewer MoE experts in the self-draft, with adaptive pruning, reuse gating, and KV sharing disabled:

```bash
./build/bin/llama-speculative "${COMMON[@]}" \
  --model-draft "$SELF_DRAFT" -ngld 99 \
  --draft 8 --draft-expert-topk 2 \
  --prune 0 --moe-reuse-strength 0 --moe-reuse-expert-cap 0
```

## LayerSkip

LayerSkip uses the same speculative command as ExpertSkip with a model-specific layer-skipped draft model. The skipped layers are part of the drafter configuration, so there is no universal mask that is correct for every architecture.

```bash
./build/bin/llama-speculative "${COMMON[@]}" \
  --model-draft /path/to/layer-skipped-draft.gguf -ngld 99 \
  --draft 8 --prune 0 \
  --moe-reuse-strength 0 --moe-reuse-expert-cap 0
```

## [EAGLE-3](https://arxiv.org/abs/2503.01840)

EAGLE-3 requires a compatible draft GGUF carrying the encoder/decoder metadata and target-layer mapping:

```bash
./build/bin/llama-speculative "${COMMON[@]}" \
  -md /path/to/eagle3-draft.gguf -ngld 99 \
  --spec-type draft-eagle3 \
  --draft 8 --draft-p-min 0.5
```

## [Cascade](https://arxiv.org/abs/2506.20675) + [EAGLE-3](https://arxiv.org/abs/2503.01840)

Enable Cascade-style test/set selection on top of the EAGLE-3 path:

```bash
./build/bin/llama-speculative "${COMMON[@]}" \
  -md /path/to/eagle3-draft.gguf -ngld 99 \
  --spec-type draft-eagle3 \
  --draft 8 --draft-p-min 0.5 \
  --moe-utility-spec \
  --spec-utility-test-iters 4 --spec-utility-set-iters 16
```

## S²-MoE

S²-MoE uses the target as a sparse self-drafter, shares KV state, enables routing-aware pruning, and derives reuse strength at runtime:

```bash
./build/bin/llama-speculative "${COMMON[@]}" \
  --model-draft "$SELF_DRAFT" -ngld 99 \
  --draft 8 --draft-share-kv --draft-expert-topk 2 \
  --moe-reuse-strength 0 --moe-reuse-expert-cap 18 --moe-reuse-runtime \
  --prune 1 --prune-max-nodes 512 --prune-max-depth 32 \
  --prune-m-route 8 --prune-k-tgt 8 --prune-score-thresh 1
```

Use the complete profiled cost-model arguments from [the speculative decoder guide](../examples/speculative/README.md).

## [DFlash](https://arxiv.org/abs/2602.06036) and [Domino](https://arxiv.org/abs/2605.29707)

DFlash requires a compatible draft GGUF with DFlash metadata. Domino uses the same runtime flag; the loader detects Domino metadata and selects its GPU sampling path automatically.

```bash
./build/bin/llama-speculative "${COMMON[@]}" \
  -md /path/to/dflash-or-domino-draft.gguf -ngld 99 \
  --spec-type draft-dflash --draft 8
```

The native EAGLE-3, DFlash, and Domino draft GGUFs are architecture-specific. A draft model built for one target family cannot be reused with another target family.
