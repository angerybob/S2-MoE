Qwen3-Next uses hybrid recurrent memory, so trimming the attention KV cache alone cannot undo draft writes or rejected verification tokens. This ports the existing Orin prototype onto the public `orin` branch and adds a dedicated `llama-speculative-next` entry point.

The draft preserves every Gated DeltaNet, full-attention, and shared-expert layer while reducing routed experts. Before verification, it restores a recurrent prefix snapshot. During verification, it captures convolution and DeltaNet states for each prefix, then commits the target state for the accepted prefix without replaying it.

CPU/CUDA matrix paths now honor the explicit F32 precision request consistently, and SSD-mode Next loading avoids eager population of the entire expert mapping.

### Scope and limitations

- Experimental greedy decoding: one request, one linear draft chain, shared hybrid memory, depth 0–31.
- CUDA graphs and graph reuse are disabled; unsupported CLI configurations fail explicitly.
- Includes model loading, state-history APIs, memory accounting, usage/cost documentation, and a CPU correctness workflow.
- Qwen3.5-MoE and `qwen35` tokenizer support are not implemented. Related to #1; this PR does not close that issue.
- No throughput claim. Quantized models, fused recurrent projections, partial/multiple GPU offload, and stochastic sampling remain unqualified.

### Validation on Jetson Orin (2026-10-17)

- CPU-only build: 9/9 F32/F16/BF16 fixture, state, and greedy-equivalence checks passed.
- CUDA build: 12/12 CPU-selected Next checks and existing MoE reuse/runtime-reuse/DFlash regressions passed.
- Explicit GPU: 6/6 state and greedy-equivalence checks passed across the three weight types.
- State checks cover all prefixes of lengths 1, 2, 4, 9, and 32, correction continuation, 64-token prefill, and rejected lifecycle/configuration misuse.
- Detailed state/logit comparisons use F32 KV; executable tests use default F16 KV.

### Extended validation

The initial four-token `Hello` smoke test has been extended with a real Qwen3-Next-80B-A3B-Thinking F16 split-expert validation matrix on Jetson Orin, using SSD-backed experts and greedy decoding:

- Three raw prompts (science, code, and math), generating 32 tokens per prompt.
- Draft depth / `draft_expert_topk` pairs `(1, 1)`, `(4, 2)`, and `(8, 4)`, compared against the corresponding autoregressive (AR) reference with the same input and decoding settings.
- **9/9 speculative token-ID sequences exactly matched their AR references.**
- Audited verification rounds include **31 zero-accept, 14 partial-accept, and 57 full-accept rounds**. Native history-copy audits and proposal/commit accounting passed; accepted-prefix replay count was zero.

The model and binary were fixed throughout this matrix. The inference source revision was `b1ba2f11e`; subsequent commits through the published implementation/evidence snapshot `75aee813d` do not change inference source. These results establish agreement under the tested conditions, not equivalence for all inputs or a throughput improvement. Qwen3.5-MoE remains unimplemented and unvalidated.

See the [per-case matrix, token IDs, and commit audits](https://github.com/angerybob/S2-MoE/blob/feature/qwen3next-self-spec/docs/validation/qwen3next-80b-20261017.json), [usage and reproduction instructions](https://github.com/angerybob/S2-MoE/blob/feature/qwen3next-self-spec/docs/qwen3next-self-spec.md), and the retained [detailed validation comment](https://github.com/angerybob/S2-MoE/pull/2#issuecomment-6232094471).

### GitHub CI and controlled type-check comparison

- The [Experimental Qwen3-Next workflow passed](https://github.com/angerybob/S2-MoE/actions/runs/39016740868).
- Repository-wide Python type-checking still fails. A [controlled baseline comparison](https://github.com/angerybob/S2-MoE/actions/runs/39019436391) checked base `orin` (`48e952cfc48862b943f409d5374802e4a771dbb5`) against candidate `b1ba2f11e` using one shared Ubuntu/Python 3.11 dependency installation, Pyright 1.1.382, and the original CI flags/configuration.
- Both revisions reported **17 errors and 1 warning**, with identical normalized diagnostics: **0 added and 0 removed**. The comparison passes because it finds no new diagnostics; the repository-wide type-check itself is not green. No unrelated script fixes are included.
- The [earlier repository CI run](https://github.com/angerybob/S2-MoE/actions/runs/39016740873) reported 10 errors in a different dependency environment. The controlled comparison is a separate reproducible result, not a reconstruction of that historical environment.

The comparison run retains raw reports and a shared dependency freeze; a [committed summary](https://github.com/angerybob/S2-MoE/blob/feature/qwen3next-self-spec/docs/validation/qwen3next-typecheck-baseline-20261017.json) records its scope and results.

### Branch and documentation

- Base: `orin` (`48e952cfc48862b943f409d5374802e4a771dbb5`)
- Feature branch: `feature/qwen3next-self-spec`
- Published implementation and validation-evidence snapshot: [`75aee813d`](https://github.com/angerybob/S2-MoE/commit/75aee813df62d2ed41b0fc086e7367406f4730cd). Later PR-description maintenance commits do not change inference code.
- Documentation: [Qwen3-Next Self-Speculation](https://github.com/angerybob/S2-MoE/blob/feature/qwen3next-self-spec/docs/qwen3next-self-spec.md)
