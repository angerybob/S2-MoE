# Orin GPU validation of the Qwen hybrid S2-MoE integration

This record qualifies the tested SSD placement, state handling and core S2-MoE
connections for Next and Qwen3.5. It does not establish universal numerical
correctness or a general speedup. See [deployment and limitations](qwen-hybrid-s2moe.md).

## GPU functional checks

Source `c0401b67b8a185bd5396415ea6aa2364aa355b03`, Jetson AGX Orin, CUDA sm87.
The CUDA build passes **34/34** CTest targets. This includes 31 host/state/CLI/
metadata tests and three actual GPU targets; it is not 34 independent GPU model
experiments. Both architecture-specific GPU suites pass **12/12** cases each:
one resident oracle plus 11 SSD cases, with 11 exact-mode checks and one
separately marked approximate-reuse run. Exact speculative fixture runs audit
prefix copies. The Next tree tests perform six alternative-child switches per
tree case; Qwen3.5 performs three. All emitted tokens match resident AR in exact
mode. All-cold, automatic/full residency, partial residency, zero draft depth,
zero-extension pruning, partial/full acceptance and tree continuation are covered.

The Q8 operator regression passes **36** bit-identical comparisons against
serial resident projections: 20 resident and 16 partially resident SSD cases,
including sorted expert batches, up/down paths, and strided top-k IDs. The SSD
path now pads temporary CUDA weights and retains serial Q8 arithmetic instead
of forcing cold pool allocations through an incompatible MMQ fallback.

## Real-model combined-path checks

The main `llama-speculative` architecture dispatcher runs both actual models on
GPU. All non-routed weights and budget-selected experts stay resident; cold
experts use direct SSD I/O. Each model is loaded once, with fresh logical state
for each case. Greedy/raw text, eight output tokens, EOS ignored, `-ngl 99`,
`-c 128 -ub 32 -t 8`, copy audit off; CUDA graphs/reuse of graphs off.

| Model | Total/reserve GiB | Resident model buffers GiB | Resident expert bundles | Layers on GPU |
|---|---:|---:|---:|---:|
| Qwen3.5-35B-A3B Q8_0 | 8 / 3 | 5.00 | 808 / 10240 | 41 / 41 |
| Qwen3-Next-80B-A3B Thinking F16 | 16 / 3 | 13.00 | 1450 / 24576 | 49 / 49 |

Each model passes AR plus exact tree/prune D1/K2 and D4/K2 comparisons. Branch
width is two; `--prune-tpot 5000 --prune-bandwidth 0.15 --prune-score-thresh 1`.
The D4 exact runs reject 13 Qwen3.5 expansions and 16 Next expansions, showing
that pruning executes. Real-model eight-token checks do not themselves trigger
alternative-child switches; those are exercised by the longer fixture cases.
The separate D4/K2 runtime-reuse case (expert cap 16) also completes with finite
outputs, resident hits and SSD reads. It is approximate and has no exact-AR
assertion. These short runs qualify execution of the combined paths, not long
generation or model quality. No accepted target prefix is replayed.

Next already had a GPU SSD prototype and recorded 9/9 32-token comparisons;
this change adds budgeted placement, packed-GGUF indexing, adaptive trees/
pruning, verification reuse and the main/config entry points to it.

The [machine-readable integration report](validation/qwen-hybrid-ssd-integration-20261018.json)
contains exact tokens, counters, copy-audit flags, placements and individual cases.

## 32-token Qwen3.5 SSD timing subset

Measured source `2f5e1ee1d`, official ggml-org Q8_0 artifact, SHA-256
`8a83fbfde74b0366feb76b075c10df46bbece6aa21ed14a6f291a7904f2b7d67`.
This is an **8 GiB allocation budget / 2 GiB runtime reserve** experiment with
6.00 GiB resident model buffers and 1129/10240 resident expert bundles. It is a
different reserve/placement from the short combined-path checks above. CPU/host
and CUDA allocations share Jetson LPDDR; the weight total includes both.
The automatic placement is an ID-balanced fallback, not a profiled hot list.

Science prompt: `Explain why the sky looks blue in two short sentences.`
All four depth/top-k configurations reproduce the same **32-token** AR output.
Audit off, quiet, direct I/O; one linear draft chain, pruning/reuse off to isolate
phase costs. Rates are 31 post-prefill tokens / generation seconds; loading,
prefill and case setup are excluded. AR before/after uses identical placement.
The `resident_model` JSON field means loaded-model reuse across cases, not full
expert residency.

| Case | Decode tokens/s | SSD payload GiB | SSD read seconds | Exact AR match |
|---|---:|---:|---:|---|
| r1-ar | 0.1534 | 27.51 | 196.84 | yes |
| r1-d1-k4 | 0.2231 | 31.42 | 132.64 | yes |
| r1-d4-k1 | 0.1321 | 36.83 | 225.38 | yes |
| r1-d1-k1 | 0.2698 | 29.10 | 108.61 | yes |
| r1-d4-k4 | 0.2428 | 29.58 | 121.67 | yes |
| final-ar | 0.2150 | 27.51 | 138.80 | yes |

The excluded AR warmup is 0.2787 tokens/s. Two measured AR brackets vary from
0.1534 to 0.2150 tokens/s despite reading the same 27.51 GiB. These are single
measurements under substantial SSD-latency variation, so no robust speedup is
claimed from their ratios. The AR comparison matches placement; it is not a
capacity-optimized AR configuration using a smaller runtime reserve.

| Speculation | Draft s | Verify + history s | Root snapshot s | Restore s | Commit s |
|---|---:|---:|---:|---:|---:|
| D1/K4 | 20.276 | 117.935 | 0.457 | 0.152 | 0.033 |
| D4/K1 | 6.805 | 227.406 | 0.351 | 0.115 | 0.025 |
| D1/K1 | 5.090 | 109.164 | 0.396 | 0.171 | 0.036 |
| D4/K4 | 40.577 | 86.847 | 0.146 | 0.066 | 0.014 |

SSD reads account for roughly 95¨C97% of post-prefill wall time in these runs.
The external snapshot/restore/commit APIs account for under 1%; in-graph history
writes remain inside verification and have not been individually kernel-timed.
Thus this memory-limited deployment is primarily I/O-bound, unlike the previous
fully resident experiment. Profiled hot-expert placement and reducing cold
expert traffic matter before further state-copy optimization.

The original broad plan contained 17 cases. It was intentionally stopped after
the seven-case main matrix completed (one warmup, two measured AR brackets,
four speculative configurations) to run the separate two-architecture integration
checks. One extra code AR warmup completed; the measured code/math confirmation
matrix did not. The [original-status timing report](validation/qwen-hybrid-ssd-main-qwen35-20261018.json)
retains its interrupted/failed parent status and explicitly marks the main
subset passed. It is not represented as a completed 17-case matrix.

## CI and remaining qualification

- [Next CPU/entry/placement CI](https://github.com/angerybob/S2-MoE/actions/runs/39108108225): passed.
- [Qwen3.5 independent AR + CPU/entry/placement CI](https://github.com/angerybob/S2-MoE/actions/runs/39108108285): passed.
- [Controlled type baseline](https://github.com/angerybob/S2-MoE/actions/runs/39108108229): passed with 17 errors / one warning in both revisions and zero new diagnostics. Diagnostic identity excludes moved source positions; raw ranges and message multiplicity remain available. The existing optional-iterable error in `run_spec.py` moved six lines when adding dispatch; it was not fixed or hidden.

These results do not make the inherited repository-wide lint/type/platform
checks green. No unrelated legacy-script repairs are included.

The previously recorded [fully resident Qwen3.5 CUDA math mismatch](qwen35-orin-gpu-benchmark.md)
at output index 17 remains an open general-correctness limitation. This change
does not complete a new long-form CUDA math qualification. PR #3 remains Draft;
sampling, multiple requests, vision/MTP, layer skipping and flat-tree hybrid
verification are outside this driver. Approximate expert reuse requires separate
quality evaluation before release claims.
