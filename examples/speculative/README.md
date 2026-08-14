# llama.cpp/examples/speculative

Demonstration of speculative decoding and tree-based speculative decoding techniques

More info:

- https://github.com/ggml-org/llama.cpp/pull/2926
- https://github.com/ggml-org/llama.cpp/pull/3624
- https://github.com/ggml-org/llama.cpp/pull/5625

## Profiled RTX 4090 S2-MoE configurations

Enable `--moe-reuse-runtime` to derive one reuse strength per parallel target
verification batch from the mean `top1 - top(k+1)` margin. The runtime policy
is disabled for prefill and ordinary decoding.

The row below is a representative profiled example rather than an exhaustive
model matrix. It uses the split-expert target itself as the drafter.

| Model | Target top-k | Draft top-k | Draft cap/min | Expert cap | TPOT | EPS | Expert MiB | BW MiB/ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-30B-A3B | 8 | 3 | 16/6 | 14 | 1121.076233 | 134.887046 | 9.000000 | 1.562478 |

Use the example row to fill the uppercase values below. `HOT_EXPERTS_JSON`
is the profiled resident-expert list for the target model.

```bash
numactl --cpunodebind=1 --membind=1 env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  GGML_CUDA_DISABLE_GRAPHS=1 LLAMA_GRAPH_REUSE_DISABLE=1 \
  LLAMA_CUDA_MOE_FORCE_STAGING=1 \
  ./build/bin/llama-speculative \
  -m "$TARGET_MODEL" -md "$TARGET_MODEL" \
  --gpu-experts-json "$HOT_EXPERTS_JSON" \
  --n-expert-used "$TARGET_TOPK" --draft-n-expert-used "$DRAFT_TOPK" \
  -f "$PROMPT_FILE" -n 64 -c 1024 -b 1024 -ub 512 -ngl "$GPU_LAYERS" \
  --temp 0 --top-p 0.95 --min-p 0.05 --seed 1 \
  --draft "$DRAFT_CAP" --draft-min "$DRAFT_MIN" --draft-p-min 0.1 \
  --draft-share-kv --moe-reuse-runtime \
  --moe-reuse-expert-cap "$EXPERT_CAP" \
  --prune 2 --prune-max-depth 32 --prune-max-nodes 256 \
  --prune-budget 1000000 --prune-score-thresh 1 --prune-conf-thresh -1 \
  --prune-tpot "$TPOT" --prune-eps "$EPS" \
  --prune-expert-bytes "$EXPERT_MIB" --prune-bandwidth "$BANDWIDTH" \
  --ignore-eos
```

The example uses `GPU_LAYERS=99`. Do not add adaptive expert caps, a
draft-expert cache, or utility-driven draft selection when reproducing it.
