`orin` has a limited Qwen3-Next prototype and does not load `qwen35moe` / the `qwen35` pre-tokenizer. This adds the Qwen3.5 architecture integration and connects **both hybrid models to the core S2-MoE GPU deployment path**: resident non-routed weights, budget-selected resident experts, SSD cold experts, expert-sparse drafts, adaptive pruning/trees, and optional verification-time expert reuse.

The main `llama-speculative` entry point now dispatches these architectures to the shared Hybrid driver. Dedicated Next/Qwen3.5 entry points remain available; the JSON runner includes explicit hybrid configuration templates.

### Implementation

- Load Qwen3.5 text-only GGUFs with separate alpha/beta projections, tiled DeltaNet key heads, stable softplus and the checked scalar-position reduction of canonical IMRoPE. Preserve every recurrent, full-attention and shared-expert layer while reducing only draft routed-expert computation.
- Index packed GGUF expert byte ranges directly, retaining existing split-expert support. Keep non-routed weights and selected whole expert bundles resident; validate the hot list, dimensions/types and quantized allocation padding. A total allocation budget explicitly reserves shared KV/recurrent state, verification history, contexts, snapshots and staging before choosing experts. It is an allocation guard, not an OS/cgroup limit.
- Execute cold experts on CUDA with aligned direct SSD I/O by default. Fix the cold-Q8 temporary-buffer fallback so sorted batches retain the same projection arithmetic as resident serial AR; fail on invalid/short expert reads.
- Add routing-aware gain/I/O-cost pruning and bounded DFS draft trees. Ancestor snapshots restore draft forks safely. Target verification is **staged by path**: commit the target-computed accepted prefix before following an alternate subtree, without replaying accepted target inputs. This differs from a flattened Transformer tree batch.
- Connect the existing two-pass expert-reuse operator **only during multi-token target verification**. It is off by default. Enabling it can change target routing/logits, so approximate-reuse execution is reported separately from exact-mode AR comparisons.

### CUDA AR-consistency fix

The recorded long-form Qwen3.5 CUDA math mismatch at output index 17 is fixed for the tested configurations. Serial AR and batched verification previously selected different floating matrix-reduction geometry even with F32 accumulation. A per-node CUDA dispatch hint now keeps the reductions consistent, retaining grouped heads and parallel token samples. It is independent of profiling/auditing. Verification remains batched and accepted target prefixes are committed without replay.

The final same-prefix diagnostic at offsets 0/8/16 has **zero difference** in both logit rows, every recurrent R/S array, the selected history state and continuation. Disabling fusion alone did not fix the original discrepancy. All new math runs also match the **unchanged pre-fix ordinary-AR token sequence**, so the reference was not changed to conceal the failure.

### Extended GPU validation

Tested inference revision: `a7128ceb96d01349fdf895de6b8c4235049af2c0` on orin-60, original ggml-org Qwen3.5-35B-A3B Q8_0. All **41/41 layers run on GPU**. Auditing and approximate expert reuse are off; raw prompts, greedy decoding and fixed output lengths are used.

| Total budget / runtime reserve | Prompts | Output length | Exact speculative settings | Completed AR + speculative cases |
|---|---|---:|---|---:|
| 24 GiB / 3 GiB | math, science, code, Chinese | 64 | D1/K2, D8/K4 | 13/13 |
| 8 GiB / 3 GiB | original math | 32 | D1/K2, D4/K2, D8/K4 | 4/4 |
| 8 GiB / 3 GiB | original math | 32 | D4/K2, branch width 2 + pruning | 2/2 |

All **19/19 SSD cases** finish: seven AR/control runs and **12/12 exact speculative matches**, totaling 1024 emitted tokens. All non-routed weights and selected experts remain resident; the 24/8 GiB budgets retain 5948/808 of 10240 expert bundles respectively. Every case verifies resident hits, actual direct SSD reads and the allocation guard. Remaining experts execute on CUDA after SSD loading. This guard is not an OS/cgroup limit.

A separate fully resident GPU reproduction/control completes 13/13 cases at 64 tokens across the same four prompts, including repeated math AR. It isolates the former numerical failure; the table above qualifies the memory-limited SSD path. Zero/partial/full acceptance, exact token arrays, commands and runtime counters are preserved in the report.

The SSD runs record 14 zero-accept, 5 partial-accept and 186 full-accept rounds. The real pruned-tree run removes 50 expansions and exercises an alternate-child switch while retaining exact AR output.

- Orin CUDA build: **35/35** CTest targets pass (31 host/state/CLI/metadata and four actual GPU targets).
- **192** F32/F16/BF16 operator comparisons and **36** Q8 resident/SSD comparisons are bit-identical to serial arithmetic.
- Existing GPU SSD fixture suites remain **12/12 per architecture**, including 11 exact-mode cases and one separately marked approximate-reuse case. The earlier real Next-80B-A3B 16 GiB integration checks remain eight-token smoke tests; this new long-output matrix is specific to Qwen3.5.

[Current GPU diagnosis, results and reproduction](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/qwen35-cuda-ar-fix.md) · [Full validation JSON](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/validation/qwen35-cuda-ar-fix-20261018.json) · [Deployment/configuration guide](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/qwen-hybrid-s2moe.md).

### Historical timing evidence

A separate Qwen3.5 run uses an **8 GiB / 2 GiB reserve** configuration (6.00 GiB resident weights; 1129 resident expert bundles), direct I/O, audit off and one linear draft branch. All four `(depth, top-k)` configurations `(1,1)`, `(1,4)`, `(4,1)`, `(4,4)` match the same **32-token** AR output. Pruning/reuse are off in this control matrix to isolate phase costs.

Observed rates are 0.2698 / 0.2231 / 0.1321 / 0.2428 decode tokens/s respectively; measured AR brackets are 0.1534 and 0.2150 (excluded warmup: 0.2787). **No robust speedup is claimed** from this single-repeat, high-variance SSD measurement. AR reads 27.51 GiB of cold expert payload. SSD reads account for roughly 95–97% of generation wall time; external snapshot/restore/commit calls account for under 1%, with in-graph history writes still included in verification.

Only the seven-case main subset completed. The original 17-case plan was intentionally stopped afterward to qualify both real architectures; one extra code warmup completed, while measured code/math confirmations did not. The parent report retains its interrupted/failed status and identifies the passed main subset. The same-placement AR baseline is not a capacity-optimized AR run with a smaller reserve.

[Deployment/configuration guide](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/qwen-hybrid-s2moe.md) · [GPU validation and phase tables](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/qwen-hybrid-orin-validation.md) · [Integration JSON](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/validation/qwen-hybrid-ssd-integration-20261018.json) · [Bounded timing report](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/validation/qwen-hybrid-ssd-main-qwen35-20261018.json).

These measurements precede the CUDA consistency fix. They are retained as historical evidence, not presented as post-fix performance results. The complete new correctness matrix above supersedes the old numerical blocker; no new speedup claim is made.

### Earlier architecture evidence and CI

The earlier independent upstream comparisons, real tokenizer checks, target-expert invariance audit and CPU real-model 9/9 × 32-token matrix remain recorded in the [Qwen3.5 architecture report](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/qwen35-hybrid-self-spec.md). They are supplementary architecture evidence and do not replace GPU qualification.

[Next CI](https://github.com/angerybob/S2-MoE/actions/runs/39113622985), [Qwen3.5 independent-reference/CPU CI](https://github.com/angerybob/S2-MoE/actions/runs/39113622889) and [controlled type baseline](https://github.com/angerybob/S2-MoE/actions/runs/39113622987) pass on the tested inference revision. The controlled baseline has 17 errors / one warning in both revisions, with zero new diagnostics. Diagnostic identity excludes shifted source positions; full original ranges and multiplicity are retained. Existing repository-wide lint/type/platform failures are not represented as green; unrelated legacy scripts are unchanged.

### Experimental scope

One request, sequence zero, shared Hybrid memory, greedy/raw-text decoding, draft depth 0–31 and history capacity at most 32. Branch width is bounded and verification is staged by path. CUDA graphs/reuse, stochastic acceptance, multiple requests, dataset/chat-template integration, vision/MTP and layer skipping remain outside this driver.

The recorded GPU blocker is resolved for the tested checkpoint/configurations; arbitrary prompts, contexts, quantizations and backends are not universally qualified. Approximate expert reuse needs separate quality evaluation. This PR remains **Draft** and makes no general speedup claim.

Related to #1. Keep the compatibility/deployment request open.
