`orin` has a limited Qwen3-Next prototype and does not load `qwen35moe` / the `qwen35` pre-tokenizer. This adds the Qwen3.5 architecture integration and connects **both hybrid models to the core S2-MoE GPU deployment path**: resident non-routed weights, budget-selected resident experts, SSD cold experts, expert-sparse drafts, adaptive pruning/trees, and optional verification-time expert reuse.

The main `llama-speculative` entry point now dispatches these architectures to the shared Hybrid driver. Dedicated Next/Qwen3.5 entry points remain available; the JSON runner includes explicit hybrid configuration templates.

### Implementation

- Load Qwen3.5 text-only GGUFs with separate alpha/beta projections, tiled DeltaNet key heads, stable softplus and the checked scalar-position reduction of canonical IMRoPE. Preserve every recurrent, full-attention and shared-expert layer while reducing only draft routed-expert computation.
- Index packed GGUF expert byte ranges directly, retaining existing split-expert support. Keep non-routed weights and selected whole expert bundles resident; validate the hot list, dimensions/types and quantized allocation padding. A total allocation budget explicitly reserves shared KV/recurrent state, verification history, contexts, snapshots and staging before choosing experts. It is an allocation guard, not an OS/cgroup limit.
- Execute cold experts on CUDA with aligned direct SSD I/O by default. Fix the cold-Q8 temporary-buffer fallback so sorted batches retain the same projection arithmetic as resident serial AR; fail on invalid/short expert reads.
- Add routing-aware gain/I/O-cost pruning and bounded DFS draft trees. Ancestor snapshots restore draft forks safely. Target verification is **staged by path**: commit the target-computed accepted prefix before following an alternate subtree, without replaying accepted target inputs. This differs from a flattened Transformer tree batch.
- Connect the existing two-pass expert-reuse operator **only during multi-token target verification**. It is off by default. Enabling it can change target routing/logits, so approximate-reuse execution is reported separately from exact-mode AR comparisons.

### GPU validation

Tested inference revision: `c0401b67b8a185bd5396415ea6aa2364aa355b03`.

- Orin CUDA build regression: **34/34** targets pass (31 host/state/CLI/metadata targets plus three actual GPU targets).
- Actual GPU fixture suites: **12/12 per architecture**, covering full/partial/no resident experts, direct SSD reads, automatic placement, zero/partial/full acceptance, zero-extension pruning, audited history commits and alternate-child switches. Each suite includes 11 exact-mode checks and one separately marked approximate-reuse case.
- CUDA Q8 projection regression: **36 bit-identical** resident/partial-resident comparisons against serial resident arithmetic, including sorted up/down expert batches and strided routing IDs.
- Real combined-path checks through the main executable: Qwen3.5-35B-A3B Q8_0 at **8 GiB**, Next-80B-A3B Thinking F16 at **16 GiB**. Each completes AR, exact D1/K2 and D4/K2 tree/prune checks, and a separate approximate runtime-reuse case. Exact-mode outputs match AR for **eight tokens**; these are short integration checks, not full equivalence or speed/quality benchmarks.

| Real deployment | Runtime reserve | Resident model buffers | Resident expert bundles | GPU layers |
|---|---:|---:|---:|---:|
| Qwen3.5, 8 GiB budget | 3 GiB | 5.00 GiB | 808 / 10240 | 41 / 41 |
| Qwen3-Next, 16 GiB budget | 3 GiB | 13.00 GiB | 1450 / 24576 | 49 / 49 |

The real D4 exact runs prune 13 Qwen3.5 and 16 Next expansions. Alternative-child switching is exercised by the longer GPU fixture cases; it is not claimed for the short real runs.

### Memory-limited GPU timing follow-up

A separate Qwen3.5 run uses an **8 GiB / 2 GiB reserve** configuration (6.00 GiB resident weights; 1129 resident expert bundles), direct I/O, audit off and one linear draft branch. All four `(depth, top-k)` configurations `(1,1)`, `(1,4)`, `(4,1)`, `(4,4)` match the same **32-token** AR output. Pruning/reuse are off in this control matrix to isolate phase costs.

Observed rates are 0.2698 / 0.2231 / 0.1321 / 0.2428 decode tokens/s respectively; measured AR brackets are 0.1534 and 0.2150 (excluded warmup: 0.2787). **No robust speedup is claimed** from this single-repeat, high-variance SSD measurement. AR reads 27.51 GiB of cold expert payload. SSD reads account for roughly 95–97% of generation wall time; external snapshot/restore/commit calls account for under 1%, with in-graph history writes still included in verification.

Only the seven-case main subset completed. The original 17-case plan was intentionally stopped afterward to qualify both real architectures; one extra code warmup completed, while measured code/math confirmations did not. The parent report retains its interrupted/failed status and identifies the passed main subset. The same-placement AR baseline is not a capacity-optimized AR run with a smaller reserve.

[Deployment/configuration guide](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/qwen-hybrid-s2moe.md) · [GPU validation and phase tables](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/qwen-hybrid-orin-validation.md) · [Integration JSON](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/validation/qwen-hybrid-ssd-integration-20261018.json) · [Bounded timing report](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/validation/qwen-hybrid-ssd-main-qwen35-20261018.json).

### Earlier architecture evidence and CI

The earlier independent upstream comparisons, real tokenizer checks, target-expert invariance audit and CPU real-model 9/9 × 32-token matrix remain recorded in the [Qwen3.5 architecture report](https://github.com/angerybob/S2-MoE/blob/feature/qwen35-hybrid-self-spec/docs/qwen35-hybrid-self-spec.md). They are supplementary architecture evidence and do not replace GPU qualification.

[Next CI](https://github.com/angerybob/S2-MoE/actions/runs/39108108225), [Qwen3.5 independent-reference/CPU CI](https://github.com/angerybob/S2-MoE/actions/runs/39108108285) and [controlled type baseline](https://github.com/angerybob/S2-MoE/actions/runs/39108108229) pass on the tested revision. The baseline has 17 errors / one warning in both revisions, with zero new diagnostics. Diagnostic identity excludes shifted source positions; full original ranges and multiplicity are retained. Existing repository-wide lint/type/platform failures are not represented as green, and unrelated legacy scripts are not repaired.

### Experimental scope and remaining blocker

One request, sequence zero, shared Hybrid memory, greedy/raw-text decoding, draft depth 0–31 and history capacity at most 32. Branch width is bounded and verification is staged by path. CUDA graphs/reuse, stochastic acceptance, multiple requests, dataset/chat-template integration, vision/MTP and layer skipping remain outside this driver.

The previously observed **fully resident Qwen3.5 CUDA math mismatch at output index 17 remains an open general-correctness limitation**; this change does not complete a new long-form CUDA math qualification. Approximate expert reuse also needs separate quality evaluation. This PR remains **Draft** and makes no general speedup or universal AR-equivalence claim.

Related to #1. Keep the compatibility/deployment request open.
