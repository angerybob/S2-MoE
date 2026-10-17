Qwen3-Next has recurrent state that cannot be rewound by trimming attention KV alone. This ports the existing Orin prototype to the public `orin` branch and provides the limited `llama-speculative-next` entry point for review.

## Scope and limitations

- Expert-only draft sparsification; all Gated DeltaNet, full-attention, and shared-expert paths remain active.
- Shared recurrent-prefix snapshots, verification-prefix convolution/DeltaNet history, and target-state commits without accepted-prefix replay.
- Explicit F32 precision through dense/expert CPU and CUDA paths; demand-paged loading for SSD-backed Next experts.
- Experimental greedy decoding: one request, one draft chain, shared hybrid memory, draft depth 0–31. CUDA graphs and graph reuse are disabled.
- Qwen3.5-MoE and the `qwen35` tokenizer are not implemented. Quantized weights, fused recurrent projections, partial/multi-GPU offload, and stochastic sampling remain unqualified.
- No throughput or speedup claim. Related to #1.

## Validation already completed on Orin

- CPU-only build: 9/9 F32/F16/BF16 fixture/state/greedy-equivalence checks passed.
- CUDA build: 12/12 CPU-selected Next checks and existing MoE reuse/runtime-reuse/DFlash regressions passed.
- Explicit GPU: 6/6 state and greedy-equivalence checks passed across all three fixture weight types.
- State checks cover all prefixes of lengths 1, 2, 4, 9, and 32, correction continuation, 64-token prefill, and unsupported lifecycle/configuration use. Detailed state/logit checks use F32 KV; executable tests use default F16 KV.
- A real split-expert Qwen3-Next-80B-A3B-Thinking model completed SSD-backed self-speculation and AR for raw prompt `Hello`, producing the same four tokens `[11, 358, 1184, 311]`. This remains a short smoke test.

## Follow-up validation in progress

- A fixed-condition real-model matrix: three raw prompts, 32 output tokens each, and draft depth/top-k combinations `(1,1)`, `(4,2)`, `(8,4)`, with token-ID equality and per-round audited state-commit accounting.
- An `orin` versus branch Pyright comparison in a single Ubuntu/Python 3.11 dependency environment, using original CI version 1.1.382, settings, and CLI flags. The existing ten type errors are not declared baseline issues until that comparison completes.

The existing Next-specific GitHub workflow passes. Repository-wide Python type-check currently fails in older scripts; unrelated fixes are excluded from this PR.

See [the implementation and verification notes](docs/qwen3next-self-spec.md). Keep this PR in Draft while broader real-model validation is completed.
