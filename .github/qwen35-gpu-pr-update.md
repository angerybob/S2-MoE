<!-- qwen35-orin-gpu-benchmark:start -->
### Orin GPU performance and remaining correctness limitation

The GPU follow-up uses the real Qwen3.5-35B-A3B Q8_0 model with **41/41 layers offloaded** and recorded GPU utilization up to **99%**. We keep weights resident, create fresh inference state for each case, disable copy auditing/token rendering, and exclude loading/prefill. Both AR and self-spec use the same experimental executable with CUDA graphs/reuse disabled.

- **Passed main GPU sweep:** all 9 depth/top-k configurations run twice, 64 output tokens each. All 22 runs (including AR references/warmup) match token IDs. AR averages **14.864 token/s**; the fastest tested self-spec configuration (depth 8/top-k 4) averages **8.597 token/s**, or **0.578x AR throughput**. Every tested main configuration is slower.
- Best configuration per-output phase means: Draft **3.524 s**, verification including history writes **3.328 s**, snapshot **0.286 s**, restore **0.086 s**, commit **0.021 s**. External state APIs total **5.37%**; in-graph history writes remain within verification. Draft costs roughly 50-60 ms per proposed token versus 67 ms per AR decode, so its savings do not cover the extra verification pass.
- **Additional code prompt:** both depth 1/top-k 4 and depth 8/top-k 4 pass 64-token equality; rates are **8.569 / 9.270 token/s** versus code AR **14.796 token/s**.
- **Remaining GPU blocker:** the math depth 1/top-k 4 confirmation diverges at output token **18**. Separate diagnostics reproduce that divergence at other depths/top-k while repeated AR contexts remain stable. The overall suite is therefore recorded as **failed: 30/32 planned cases completed, 29 valid**. This blocks general GPU AR-equivalence qualification; the contribution remains experimental and in Draft.

GPU testing also exposed an earlier Q8 projection dispatch mismatch. Commit `8b2a294d7` adds an opt-in hybrid CUDA compatibility path; its 20 dense/expert projection regression cases are bit-identical to their serial oracle. This resolves the reproduced main-prompt mismatch, **not** the remaining math failure. Ordinary CUDA callers without the opt-in retain their existing dispatch.

[GPU matrix, full phase timings, raw evidence, numerical fix and unresolved limitation](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/qwen35-orin-gpu-benchmark.md). Timed source: `55a46245457b218c3f6cd3246fe51a9f18e63895`. The older CPU measurements below are supplementary.
<!-- qwen35-orin-gpu-benchmark:end -->
