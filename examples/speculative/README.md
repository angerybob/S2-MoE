# llama.cpp/examples/speculative

Demonstration of speculative decoding and tree-based speculative decoding techniques

More info:

- https://github.com/ggml-org/llama.cpp/pull/2926
- https://github.com/ggml-org/llama.cpp/pull/3624
- https://github.com/ggml-org/llama.cpp/pull/5625

## Profiled RTX 4090 S2-MoE configurations

Enable `--moe-reuse-runtime` to derive one reuse strength per parallel target
verification batch from the mean `top1 - top(k+1)` margin. The runtime policy
is disabled for prefill and ordinary decoding. Keep
`--prune-score-thresh 1` and apply these factors to the locally measured
pruning costs:

| Model | Draft | EPS scale | TPOT scale | Expert-size scale |
| --- | ---: | ---: | ---: | ---: |
| DeepSeek-V2-Lite | 6 | 1.00 | 1.25 | 1.00 |
| OLMoE | 9 | 0.50 | 1.00 | 1.00 |
| Qwen3-30B-A3B | 16 | 0.50 | 1.25 | 1.00 |
| GPT-OSS-120B | 6 | 1.00 | 1.25 | 1.00 |
