# llama.cpp/examples/speculative

Demonstration of speculative decoding and tree-based speculative decoding techniques

More info:

- https://github.com/ggml-org/llama.cpp/pull/2926
- https://github.com/ggml-org/llama.cpp/pull/3624
- https://github.com/ggml-org/llama.cpp/pull/5625

## Profiled Jetson AGX Orin S2-MoE configurations

Runtime reuse computes one strength per target-verification batch as the mean
`top1 - top(k+1)` margin. It is disabled during prefill, draft decoding, and
ordinary one-token decode.

The values below are the final absolute reproduction settings. A row is keyed
only by model and memory budget: use the same row for every dataset, including
RG, SU, TR, and QA. Dataset selection changes only the input.

`self` means that `--model-draft` is the same split-expert GGUF as `-m`.
`separate Q4` is used only when the target is fully resident. OLMoE is fully
resident at 32 GB, so the same run also represents 64 GB.

| Model | Memory | Residency | Drafter | Draft top-k | Draft cap | Expert cap | m/k | TPOT | EPS | Expert MiB | BW MiB/ms | Expert max |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DeepSeek-V2-Lite | 16G | SSD, no resident list | self | 1 | 6 | 18 | 6/6 | 64.2 | 23.0 | 0.4544 | 2.84 | 384 |
| DeepSeek-V2-Lite | 32G | SSD + 32G resident list | self | 1 | 12 | 18 | 6/6 | 64.2 | 23.0 | 0.4544 | 2.84 | 384 |
| DeepSeek-V2-Lite | 64G | target fully resident | separate Q4 | 1 | 6 | 18 | 6/6 | 64.2 | 23.0 | 32.7680 | 204.8 | 384 |
| OLMoE | 16G | SSD + 16G resident list | self | 2 | 8 | 18 | 8/8 | 100.0 | 7.5 | 0.8578125 | 3.66 | 384 |
| OLMoE | 32G/64G | target fully resident | separate Q4 | 2 | 6 | 18 | 8/8 | 100.0 | 18.75 | 32.0000 | 204.8 | 384 |
| Qwen3-30B-A3B | 16G | SSD, no resident list | self | 3 | 8 | 14 | 8/8 | 2500.0 | 25.0 | 0.9000 | 2.50 | 5840 |
| Qwen3-30B-A3B | 32G | SSD + 32G resident list | self | 3 | 6 | 14 | 8/8 | 2500.0 | 50.0 | 0.6000 | 2.50 | 5840 |
| Qwen3-30B-A3B | 64G | SSD + 64G resident list | self | 3 | 6 | 14 | 8/8 | 2500.0 | 50.0 | 0.6000 | 2.50 | 5840 |
| GPT-OSS-120B | 16G | SSD + 16G resident list | self | 1 | 4 | 6 | 4/4 | 2000.0 | 80.0 | 0.46875 | 3.00 | 5840 |
| GPT-OSS-120B | 32G | SSD + 32G resident list | self | 1 | 8 | 6 | 4/4 | 2000.0 | 20.0 | 0.46875 | 3.00 | 5840 |
| GPT-OSS-120B | 64G | SSD + 64G resident list | self | 1 | 4 | 6 | 4/4 | 2000.0 | 80.0 | 0.46875 | 3.00 | 5840 |

Use the selected row to fill the uppercase values below. For SSD rows,
`RESIDENCY_ARGS` is `--ssd-moe` plus `--hot-experts HOT.json` when the row has
a resident list. For fully resident rows it is empty.

```bash
./build/bin/llama-speculative \
  -m "$TARGET_MODEL" --model-draft "$DRAFT_MODEL" $RESIDENCY_ARGS \
  --dataset "$DATASET_JSONL" --n-questions "$N_QUESTIONS" \
  -n 64 --ignore-eos -c 2048 --temp 1 --min-p 0.01 --top-k 4 -s 42 \
  -ngl 99 -ngld 99 -t 8 --parallel 1 -kvu --no-warmup --no-mmap \
  --draft "$DRAFT_CAP" --draft-p-split 0.1 --draft-share-kv \
  --draft-expert-topk "$DRAFT_TOPK" \
  --moe-reuse-strength 0 --moe-reuse-expert-cap "$EXPERT_CAP" \
  --moe-reuse-runtime \
  --prune 1 --prune-max-nodes 512 --prune-max-depth 32 --prune-budget 0 \
  --prune-m-route "$M_ROUTE" --prune-k-tgt "$K_TARGET" \
  --prune-beta 0.8 --prune-gamma 2 --prune-lambda 1 \
  --prune-tpot "$TPOT" --prune-eps "$EPS" \
  --prune-expert-bytes "$EXPERT_MIB" --prune-bandwidth "$BANDWIDTH" \
  --prune-expert-max "$EXPERT_MAX" --prune-score-thresh 1
```

Do not add adaptive expert caps, a draft-expert cache, or utility-driven draft
selection when reproducing these settings.
