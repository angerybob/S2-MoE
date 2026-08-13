# llama.cpp/examples/speculative

Demonstration of speculative decoding and tree-based speculative decoding techniques

More info:

- https://github.com/ggml-org/llama.cpp/pull/2926
- https://github.com/ggml-org/llama.cpp/pull/3624
- https://github.com/ggml-org/llama.cpp/pull/5625

## Profiled S2-MoE configurations

`run_spec.py` accepts `moe.reuse_runtime: true`. Runtime reuse computes one
strength per target-verification batch as the mean `top1 - top(k+1)` margin.
It is disabled during prefill, draft decoding, and ordinary one-token decode.

The following scale factors are the profiled reproduction settings. Apply the
scales to the measured `prune-eps`, `prune-tpot`, and `prune-expert-bytes` on
the target machine. Keep `--prune-score-thresh 1`; do not enable
`--moe-utility-spec`.

| Hardware | Model | Draft | EPS scale | TPOT scale | Expert-size scale |
| --- | --- | ---: | ---: | ---: | ---: |
| Jetson AGX Orin 16G | DeepSeek-V2-Lite | 6 | 1.00 | 1.00 | 1.00 |
| Jetson AGX Orin 32G | DeepSeek-V2-Lite | 12 | 1.00 | 1.00 | 1.00 |
| Jetson AGX Orin 64G | DeepSeek-V2-Lite | 6 | 1.00 | 1.00 | 1.00 |
| Jetson AGX Orin 16G | OLMoE | 8 | 0.50 | 1.00 | 1.50 |
| Jetson AGX Orin 32G/64G | OLMoE | 6 | 1.00 | 1.00 | 1.00 |
| Jetson AGX Orin 16G | Qwen3-30B-A3B | 8 | 0.50 | 1.00 | 1.50 |
| Jetson AGX Orin 32G/64G | Qwen3-30B-A3B | 6 | 1.00 | 1.00 | 1.00 |
| Jetson AGX Orin 16G | GPT-OSS-120B | 4 | 2.00 | 1.00 | 1.00 |
| Jetson AGX Orin 32G | GPT-OSS-120B | 8 | 0.50 | 1.00 | 1.00 |
| Jetson AGX Orin 64G | GPT-OSS-120B | 4 | 2.00 | 1.00 | 1.00 |
| RTX 4090 | DeepSeek-V2-Lite | 6 | 1.00 | 1.25 | 1.00 |
| RTX 4090 | OLMoE | 9 | 0.50 | 1.00 | 1.00 |
| RTX 4090 | Qwen3-30B-A3B | 16 | 0.50 | 1.25 | 1.00 |
| RTX 4090 | GPT-OSS-120B | 6 | 1.00 | 1.25 | 1.00 |

For Orin memory-constrained runs, retain the memory-specific resident-expert
files and expert caps used by the model package.
