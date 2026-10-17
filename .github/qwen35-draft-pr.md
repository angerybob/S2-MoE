`orin` currently rejects `qwen35moe` models and the `qwen35` pre-tokenizer. This adds a separate experimental `llama-speculative-qwen35` entry point for the recorded text-only Qwen3.5-MoE GGUF layouts, building on the Qwen3-Next integration merged in #2.

The draft reduces routed experts while preserving every recurrent, full-attention and shared-expert path. Recurrent snapshots undo draft writes; verification-prefix history commits the target-computed accepted state without replay. The architecture adaptation uses separate alpha/beta projections, tiled key heads, stable softplus and a checked text-only reduction of canonical IMRoPE.

### Staged validation

All three stages were checked separately before enabling the final self-spec path:

- **Model/tokenizer/AR:** 12/12 real-vocabulary tokenizer comparisons; 6/6 independent upstream-vs-candidate 32-token AR comparisons (three prompts, batch/serial prefill). Upstream source is pinned to `650bf14eb9a922de0f88c9523a271159cc5ae469`.
- **Expert-only draft:** real 35B-A3B model retains all 30 recurrent, 10 full-attention and 40 shared-expert paths. Top-k 1/2 reduces routed projection size to 1/8 and 1/4 of top-k 8. Setting draft top-k on a target context leaves its logits exactly unchanged.
- **Snapshots/history:** Orin CPU and an independent x86 CPU build each pass 19/19 combined Next/Qwen3.5 checks. Explicit Orin CUDA state and greedy-loop tests pass 6/6 across F32/F16/BF16 fixtures. These cover every prefix of verification lengths 1, 2, 4, 9 and 32, correction continuation, root-only acceptance, 64-token prefill and unsupported-state/configuration rejection.

### Real-model extended validation

The CPU matrix uses Qwen3.5-35B-A3B Q8_0 converted from an existing HF snapshot with the pinned upstream converter (model SHA-256 `f23b26ea23f30a491728b1e4448f22e1bcbe69805bba8618f2277f18ac3cdaea`).

- Three raw prompts, 32 generated tokens each, draft depth/top-k `(1,1)`, `(4,2)`, `(8,4)`, EOS ignored for fixed-length comparisons: **9/9 exact self-spec/AR token-ID matches**.
- Audited rounds: **21 zero-accept, 8 partial-accept, 65 full-accept**. Native copy/accounting audits pass; accepted-prefix replay count is **0**.
- All three product AR sequences also equal the independently built upstream reference.
- A separate Orin CPU smoke test on the hash-verified original ggml-org Q8_0 model produces the same 8-token AR/self-spec sequence, covering zero/full acceptance and audited state commits. This short Orin run is distinct from the longer x86 CPU matrix.

Implementation/real-matrix revision: `d7e1ce53aaeb540435d52030d56d7ab398315723`. Later report and PR-maintenance changes do not change inference behavior.

[Full results, usage and reproduction](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/qwen35-hybrid-self-spec.md) · [real self-spec matrix](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/validation/qwen35-real-selfspec-20261017.json) · [upstream AR comparison](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/validation/qwen35-real-ar-20261017.json) · [Orin smoke](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/validation/qwen35-orin-real-smoke-20261018.json).

### Scope and costs

Experimental: one request, sequence zero, one linear shared-memory draft chain, greedy decoding, raw text, draft depth 0–31/history capacity 1–32. CUDA graphs and graph reuse are disabled. SSD/hot-expert loading is explicitly rejected. Vision/MTP, stochastic decoding, tree/multi-request speculation, other quantized model formats, fused recurrent projections, full-model CUDA and 32 GB Orin qualification remain separate work.

Measured recurrent history is 565.3125 MiB at capacity 9; each prefix commit copies 62.8125 MiB. Snapshots, compatibility-graph buffers and audit readbacks add separate costs. These checks do not establish universal equivalence or throughput improvement; **no speedup claim is made**.

[Experimental Qwen3.5 workflow passed on the implementation revision](https://github.com/angerybob/S2-MoE/actions/runs/39051393156). The inherited repository-wide platform/type-check failures are not represented as green; this PR does not mix in unrelated legacy-script or platform repairs.

Related to #1. Keep that request open for remaining compatibility and deployment qualification. This PR is a Draft for review.
