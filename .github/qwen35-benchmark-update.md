<!-- qwen35-orin-benchmark:start -->
### Orin CPU performance follow-up (auditing disabled)

We measured the original Qwen3.5-35B-A3B Q8_0 model on Orin with CPU placement, 8 threads, 64 output tokens, greedy decoding, state-copy auditing off and token rendering disabled. Loading and prompt prefill are excluded; throughput counts the 63 post-prefill decode tokens. Nine depth/top-k configurations each run twice, with an excluded AR warmup and three measured AR references.

- **AR: 5.716 token/s. Fastest tested self-spec: 3.685 token/s** (depth 1/top-k 4), or **0.645x AR throughput**. All nine configurations are slower on this recorded CPU path; all 22 main runs pass token equality.
- The fastest configuration accepts **100%** of proposals. Its mean phase costs per 64-token output are Draft **5.212 s**, verification including history writes **10.212 s**, snapshot **0.976 s**, restore **0.219 s**, and commit **0.233 s**. Separately timed state APIs total **1.428 s**, or **8.35%**; in-graph history writes remain included in verification.
- On additional code/math prompts, the selected configuration reaches **3.709 / 3.714 token/s**, compared with **5.711 / 5.698 token/s** for AR. All confirmation runs also match their AR token IDs.

Draft costs approximately 154-164 ms per proposed token versus 175 ms per AR target decode. The recurrent verification compatibility path still constructs dependent per-token steps. These measurements identify expensive Draft plus insufficient verification amortization as the main phase-level bottleneck; individual operator shares have not been measured. They establish no CUDA or SSD performance result.

[Full matrix, separate phase timings, raw measurements and reproduction](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/qwen35-orin-cpu-benchmark.md). Timed implementation: `3d69a2af8f5c997b8be49012318a23f9631a4f45`; its CPU regressions (19/19), tiny profiling checks (7/7), and dedicated push/PR CI passed. The feature remains experimental.
<!-- qwen35-orin-benchmark:end -->
