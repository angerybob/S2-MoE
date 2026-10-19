# Orin GPU validation of the Qwen hybrid S2-MoE integration

Integration checks at the revisions named below. For current
Qwen3.5 numerical qualification, see the
[CUDA AR-consistency fix and completed 32–64-token SSD matrix](qwen35-cuda-ar-fix.md).

This record covers SSD placement, state handling and core S2-MoE
connections for Next and Qwen3.5. See the [deployment guide](qwen-hybrid-s2moe.md).

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
