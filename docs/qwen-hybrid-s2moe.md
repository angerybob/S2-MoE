# Experimental S2-MoE for Qwen hybrid models

Both `qwen3next` and `qwen35moe` now connect to the S2-MoE SSD placement,
expert-sparse draft, adaptive pruning and verification-time expert-reuse paths.
The main `llama-speculative` executable detects these architectures before
entering the Transformer decoder. The dedicated `llama-speculative-next` and
`llama-speculative-qwen35` names remain available and use the same driver.

This is an experimental, one-request, raw-text, greedy implementation. It does
not qualify every model/precision/backend combination. In particular, the
[previous fully resident Qwen3.5 CUDA math mismatch](qwen35-orin-gpu-benchmark.md)
must not be hidden by the new SSD integration. Exact-mode and approximate
expert-reuse results are reported separately.

## Placement and memory accounting

All non-routed weights, including recurrent/attention layers, routers, shared
experts, embeddings and output weights, stay resident. On Jetson, CPU/host and
CUDA allocations share the same physical LPDDR; host embedding lookup does not
mean expert computation falls back to CPU. All hybrid SSD compute layers must
be offloaded with `-ngl 99`. SSD inference requires CUDA.

`--memory-budget-mib B --runtime-reserve-mib R` gives the weight loader `B-R`
MiB. It subtracts non-routed weights and selects whole routed-expert bundles
(gate/up/down together) within the remaining weight budget. An optional
`--hot-experts HOT.json` supplies a separately profiled list in the existing
`{"gpu_experts": [[layer, expert], ...]}` format. An oversized or missing
expert fails loading. Without a list, selection is deterministic and balanced
across layers by expert ID; it is a fallback, not a trained placement policy.

The driver checks allocated model, shared KV/recurrent state, verification
history, both contexts' compute/output buffers, host snapshots, and staging
allowances against the reserve and total budget. Shared state/weights are
counted once. Tree mode additionally reserves an ancestor snapshot per possible
fork depth. Increase `R` if the runtime check fails; fewer experts then fit.
The option is an allocation budget with headroom, not a Linux cgroup limit or
a bound on every driver allocation and reclaimable filesystem page.

Ordinary packed GGUF expert tensors are indexed by byte offset in place;
creating a second split-expert GGUF is unnecessary. Existing per-expert split
GGUFs are also supported. Metadata validation checks dimensions, projection
types, duplicates, and requested residency. Cold weights have no model-buffer
allocation. Resident-expert count is independent of `--draft-expert-topk`, which
controls computation per token.

SSD runs disable mmap loading and use aligned `O_DIRECT` cold-expert reads by
default. They do not flush global caches. `--ssd-buffered-io` explicitly permits
buffered reads on filesystems without direct I/O; their timings include possible
page-cache hits and must be labeled accordingly. The temporary CUDA copy has
backend-required padding. Q8 hybrid projections use the same vector arithmetic
as resident serial AR, including sorted batches of cold experts.

## Adaptive drafts and Hybrid state

`--prune 1` selects expansions by cumulative token confidence and estimated
incremental cold-expert I/O cost. Resident experts cost zero SSD bytes. Routing
capture predicts the target's original top-k even when draft computation uses
fewer experts. `--prune-budget` is an I/O-time estimate in milliseconds;
`--prune-bandwidth` is MiB/ms, `--prune-tpot` is milliseconds, and
`--prune-score-thresh` is a dimensionless expected-gain/cost threshold.
Calibrate time/bandwidth values for the deployment; template values are examples.
Depth, node count and confidence also bound expansion.

`--draft-branches 1` is the linear fast path. Values 2–8 enable bounded trees;
`--parallel` / `-np` are aliases for this branch width, not request batching.
`--prune-max-nodes` limits proposed nodes, excluding the known root (at most
512). Each root-to-leaf path remains at most 31 draft tokens and fits the
32-row verification-history limit. Nodes are explored in deterministic
confidence order using DFS, with alternatives reserved before descending.

Hybrid verification is **staged by path**, not one Transformer-style flattened
tree batch. After a mismatch that matches an alternative child, the driver
commits the target-computed accepted prefix and verifies that child's subtree
from the committed state. Accepted target prefixes are never replayed. Draft
forks restore ancestor snapshots and trim attention tails before changing
branches. An arbitrary best-first switch with only recurrent snapshots would
retain the wrong attention prefix, so it is not used. This scheduling changes
the performance tradeoff from the original Transformer tree implementation.

## Verification-time expert reuse

`--moe-reuse-runtime --moe-reuse-expert-cap N` enables the repository's existing
two-pass route bias during multi-token target verification only. Static
`--moe-reuse-strength` is also available. Prefill, draft execution and ordinary
AR remain unbiased. A zero cap disables reuse (the default).

Reuse can change which target experts execute and therefore change target
logits/output. It is an approximate S2-MoE mode; its successful execution is not
an exact AR-equivalence test. Keep it off for the exact-mode reference and
report its quality/acceptance separately when enabled.

## Running

```sh
cmake --build build-next --target llama-speculative llama-speculative-next \
    llama-speculative-qwen35 test-hybrid-placement test-cuda-moe-batch -j4

build-next/bin/llama-speculative -m /path/to/Qwen3.5-35B-A3B-Q8_0.gguf \
    --ssd-moe --memory-budget-mib 8192 --runtime-reserve-mib 3072 \
    -ngl 99 -c 128 -ub 32 -t 8 -n 32 --ignore-eos \
    -p 'Explain why the sky looks blue in two short sentences.' \
    --draft-max 8 --draft-expert-topk 2 --draft-branches 2 \
    --prune 1 --prune-max-nodes 31 --prune-max-depth 8 \
    --prune-tpot 100 --prune-bandwidth 1 --prune-score-thresh 1 \
    --timings-file timing.json --tokens-file tokens.json
```

Use the same model/placement flags with `--ar` for a controlled comparison.
For a capacity-optimized AR baseline, reduce its runtime reserve only after
checking its actual buffer usage; it can then retain additional experts under
the same total budget. Always record whether placement or only total capacity
was matched. Do not compare an SSD run with an unreported full-residency run.

The existing JSON runner accepts `"engine": "hybrid"`; see
[`spec_config_qwen35_hybrid.json`](../examples/speculative/spec_config_qwen35_hybrid.json)
and [`spec_config_qwen3next_hybrid.json`](../examples/speculative/spec_config_qwen3next_hybrid.json).
Replace the model path and optionally add a profiled hot-expert list. Both
templates keep approximate expert reuse off. The adapter rejects unsupported
fields. Dataset/chat-template, sampling, vision, MTP, layer skipping, multiple
requests and CUDA graph reuse have not been ported to this driver.

## Validation and timing

`test-hybrid-placement` checks original packed offsets/types, the non-routed
weight floor, whole-expert budget selection and oversized hot lists.
`test-cuda-moe-batch` additionally compares partial-resident Q8 sorted expert
batches against a resident serial oracle, including strided routing IDs.
`tests/test-hybrid-ssd.py` exercises actual CUDA SSD/direct-I/O execution,
partial residency, zero/partial/full draft acceptance, pruning, branch
switches and separately marked approximate reuse. It never downloads weights.

For actual GPU timings without copy audits:

```sh
python3 tests/benchmark-qwen-hybrid.py --binary build-next/bin/llama-speculative-qwen35 \
    --model /path/to/Qwen3.5-35B-A3B-Q8_0.gguf --output /path/to/new-results \
    --ssd-moe --memory-budget-mib 8192 --runtime-reserve-mib 2048 \
    --resident-model --tokens 32 --depths 1 4 8 --experts 1 2 4
```

The JSON separates prefill from generation and draft, verification (including
history writes), root snapshot/restore, and accepted-prefix commit. Tree fork
snapshot/restore times are subcomponents of `draft_ms`; `draft_compute_ms`
subtracts them. SSD counters separate draft and target payload bytes/read time,
plus resident hits; payload bytes are not filesystem block/physical-drive bytes.
Each case gets fresh inference state. Token-mismatching runs are invalid for
speed comparisons, and a failed matrix remains failed in its report.
