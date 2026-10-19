# Post-fix Qwen3.5 GPU/SSD speed measurements

Inference source: `427a6924ba1b9831b0ee5f06b741b3ed31bfc7db`; same CUDA binaries/libraries as
the [AR-consistency qualification](qwen35-cuda-ar-fix.md). Model: original
Qwen3.5-35B-A3B Q8_0. On orin-60 all 41/41 layers run on CUDA; non-routed
weights and selected experts stay resident, and cold experts use direct SSD I/O.

**Measurement quality: 181 NVMe timeout/error events were recorded
during the experiment. These are descriptive timings on an affected storage
path, not evidence of stable algorithmic speedup.** The observed
`timeout, completion polled` messages are consistent with a completion-notification
problem; the exact cause on this host is not established. NVIDIA documents a
[similar Jetson PCIe/NVMe interrupt issue](https://forums.developer.nvidia.com/t/nvme-timeout-happened-after-upgrading-from-jetpack-5-1-to-jetpack-6/296712).
The host runs `5.15.148-tegra`; all NVMe queue interrupt counters are concentrated
on CPU0. Device-wide event overlap is annotated conservatively for each case.
An absence of an overlapping message does not prove a fault-free sample.
No nominal 30-second penalties are subtracted from any timing, because events
can overlap and cannot be attributed uniquely to a process or generation step.
SMART access was unavailable to the current account, so this record makes no
claim about media health or a proven device-firmware failure.

The complete plan finishes **58/58** cases: eight excluded full-length AR
warmups and 50 measured runs, including **28/28 exact speculative matches**.
Both prompts produce 32 tokens; all math cases, including the larger-residency
AR controls, also equal the unchanged pre-fix AR prefix. Auditing, logit tracing,
batch diagnostics and approximate expert reuse are off. CUDA graphs/reuse remain
disabled. Each invocation loads weights once, with fresh contexts per case.

The executed harness is preserved at commit
[`a88614c633`](https://github.com/angerybob/S2-MoE/blob/a88614c633ca8040f0ca5e1968f7e68c3e965619/tests/benchmark-qwen35-ssd-speed.py);
its SHA-256 is recorded in the JSON. The review follow-up changes only progress
messages from `print` to logging to satisfy the repository lint plugin. It does
not change the schedule, commands, output files, validity checks, or inference
code, and does not regenerate or relabel the historical measurements.

## Comparison design

Primary total allocation budget is **8 GiB**, with **3 GiB runtime reserve**;
this keeps about 5 GiB of model buffers / 808 of 10240 expert bundles resident.
The six D/K combinations below run twice with pruning off and twice with it on.
The block order is off/on/on/off, with configuration order reversed in the second
round. Three measured AR brackets surround each six-configuration block; a
separate warmup is excluded. Pruning uses fixed `--prune-tpot 5000
--prune-bandwidth 0.15 --prune-score-thresh 1`, one branch. These are recorded
heuristic settings, not automatic calibration. The science prompt separately
repeats D4/K2 twice for each pruning mode.

AR capacity controls run before and after the matrix, with the **same 8 GiB
total budget but only 512 MiB runtime reserve**. Their model/runtime allocation
checks pass. They use more resident experts than the shared-placement controls;
the placement and exact commands are in the JSON. This avoids attributing an
unnecessarily large AR reserve to a speculative algorithm advantage.

Rates mean **31 post-prefill decode token intervals / generation wall time**
for 32 emitted tokens. Prefill, model loading, setup and warmups are excluded.
The tables aggregate two observations per setting using total tokens / total
wall time; observed ranges are not statistical confidence intervals.
Same-placement speedups use neighboring AR brackets. Capacity-AR ratios use
the four capacity-control observations, which are not adjacent to every case.

## Measured rates

Capacity AR: **0.267 token/s**, observed range
0.189–0.490, n=4.
Shared-placement math AR, pooled across all 12 measured brackets:
**0.292 token/s**, observed range
0.153–0.584.
All individual AR brackets, repeat ratios and range bounds are retained in the
[machine-readable report](validation/qwen35-ssd-postfix-speed-20261018.json).

| Configuration | Pruning | Decode token/s | Observed range | vs bracket AR | vs capacity AR | Pruned expansions, two runs |
|---|---|---:|---:|---:|---:|---:|
| D1/K1 | off | 0.355 | 0.219–0.941 | 1.48x | 1.33x | 0 |
| D1/K4 | off | 0.194 | 0.146–0.289 | 0.80x | 0.73x | 0 |
| D4/K1 | off | 0.165 | 0.131–0.221 | 0.68x | 0.62x | 0 |
| D4/K2 | off | 0.189 | 0.134–0.319 | 0.79x | 0.71x | 0 |
| D8/K2 | off | 0.342 | 0.274–0.453 | 1.42x | 1.28x | 0 |
| D8/K4 | off | 0.248 | 0.209–0.307 | 1.03x | 0.93x | 0 |
| D1/K1 | on | 0.361 | 0.335–0.392 | 1.12x | 1.35x | 0 |
| D1/K4 | on | 0.275 | 0.265–0.286 | 0.63x | 1.03x | 0 |
| D4/K1 | on | 0.267 | 0.208–0.375 | 0.61x | 1.00x | 28 |
| D4/K2 | on | 0.529 | 0.503–0.558 | 1.64x | 1.98x | 20 |
| D8/K2 | on | 0.197 | 0.185–0.211 | 0.61x | 0.74x | 20 |
| D8/K4 | on | 0.188 | 0.147–0.262 | 0.43x | 0.70x | 12 |

Science D4/K2 confirmation:

- Pruning off: 0.172 token/s, 0.48x neighboring AR; repeat ratios 0.46x, 0.50x. Its AR brackets average 0.292 token/s, observed range 0.176–1.029 (n=3).
- Pruning on: 0.122 token/s, 0.49x neighboring AR; repeat ratios 0.30x, 0.91x. Its AR brackets average 0.237 token/s, observed range 0.168–0.311 (n=3).

## Interpretation

The fastest tested aggregate math setting is D4/K2,
pruning on: 0.529 token/s,
1.64x its neighboring AR brackets and
1.98x the pooled capacity AR. Its two individual
paired ratios are 1.53x, 1.74x; faster than both adjacent
AR brackets in both repetitions: **no**.
This is the fastest setting in a tested matrix, not a general speedup result.
In particular, differing storage stalls can create a large apparent ratio even
when speculative I/O volume is similar to AR. Resolve the storage/interrupt
condition and repeat the comparisons before attributing these ratios to S2-MoE.

A concrete first-round example is D1/K1: both runs accept
13 of 18 proposed tokens,
and neither prunes an expansion. SSD payloads are
30.01 and
30.12 GiB, yet `pread` takes
26.92 and
72.59 seconds. Conservative nearby
device-event counts are 0 and
1. Nearly equal I/O work does not
yield a stable service time on this run.

Across all measured speculative runs, SSD `pread` occupies
**95.1%** of generation wall time, while external
snapshot/restore/commit calls occupy **0.25%**.
The remaining time is not all GPU arithmetic. Pruning must be judged with its
read volume, target batch count and AR-normalized comparison, rather than
the number of removed nodes alone. Those per-setting comparisons are in the JSON.
Of the 12 math settings, 0 reduce total cold-expert payload
below neighboring AR. The fastest observed setting reads
1.10x AR's payload. These I/O-work ratios
are separate from the fluctuating time needed to service those reads.

For D8/K2, pruning removes 10
extensions per run, but target verification batches increase from
5.0 to 11.0.
Total payload changes from 29.30 to
30.97 GiB. In this configuration, shorter drafts lose
some target-batch expert sharing, offsetting the saved Draft work. The fixed
heuristic is not established as an optimal pruning policy for this workload.

## Phase costs

Seconds per 32-token run, averaged across two observations. In the Draft and
verification columns, parentheses show SSD `pread` time within that phase.
The copy column shows snapshot/restore/commit components in seconds. SSD `pread` is
**nested inside** Draft/verification/AR wall time and must not be added again.
External copies are snapshot + restore + accepted-prefix commit. History
writes inside the graph remain included in verification time. These counters
do not isolate pure GPU arithmetic or in-graph history-copy cost.

| Configuration | Pruning | Total s | Draft s (SSD) | Verify + history s (SSD) | Copies s (snapshot/restore/commit) | SSD payload GiB |
|---|---|---:|---:|---:|---:|---:|
| D1/K1 | off | 87.32 | 5.22 (3.97) | 81.54 (76.95) | 0.542 (0.337/0.170/0.035) | 30.01 |
| D1/K4 | off | 159.89 | 52.40 (50.76) | 106.99 (102.99) | 0.483 (0.300/0.151/0.031) | 32.64 |
| D4/K1 | off | 188.16 | 6.12 (3.38) | 181.72 (176.16) | 0.302 (0.187/0.094/0.020) | 34.86 |
| D4/K2 | off | 164.15 | 53.18 (50.83) | 110.71 (106.70) | 0.241 (0.150/0.075/0.016) | 30.07 |
| D8/K2 | off | 90.75 | 23.71 (20.96) | 66.87 (63.14) | 0.151 (0.094/0.047/0.010) | 29.30 |
| D8/K4 | off | 124.79 | 47.17 (44.28) | 77.48 (74.48) | 0.121 (0.075/0.038/0.008) | 30.44 |
| D1/K1 | on | 85.88 | 2.85 (1.52) | 82.48 (77.97) | 0.536 (0.332/0.170/0.035) | 30.12 |
| D1/K4 | on | 112.77 | 12.56 (10.89) | 99.70 (95.44) | 0.478 (0.296/0.151/0.031) | 32.34 |
| D4/K1 | on | 115.95 | 5.64 (3.02) | 109.84 (105.43) | 0.450 (0.278/0.142/0.030) | 31.66 |
| D4/K2 | on | 58.56 | 11.95 (9.33) | 46.23 (42.55) | 0.358 (0.222/0.113/0.023) | 31.24 |
| D8/K2 | on | 157.40 | 10.14 (7.48) | 146.91 (142.72) | 0.329 (0.203/0.104/0.022) | 30.97 |
| D8/K4 | on | 164.74 | 43.96 (40.53) | 120.52 (116.94) | 0.239 (0.148/0.075/0.016) | 35.42 |

The JSON also splits SSD reads into Draft and Target, records snapshot,
restore and commit individually, and gives non-pread residuals. Those residuals
include transfers, routing, open/close, kernel launches, synchronization and
history writes; they are not pure GPU-compute measurements. Payload bandwidth
excludes direct-I/O alignment padding and open/close overhead.
The raw native `draft_compute_ms` name also includes SSD I/O and control work;
it subtracts only branch-fork copies and is not a pure CUDA-compute timer.

The pooled shared-placement AR run averages 106.23 s,
of which 101.54 s is `pread`
(95.6%), reading 28.50 GiB.
The capacity AR run averages
116.03 s, including
111.33 s `pread` (95.9%),
reading 26.40 GiB. The JSON includes per-setting
payload/request ratios against neighboring AR; those quantify I/O work separately
from fluctuating SSD service time.

## Scope and reproduction

Warm-model request costs (fresh contexts), seconds. These remain affected by
the same storage faults. Loading the model once per invocation is excluded from
this table and retained separately in every native timing record.

| Setting | Context setup s | Prefill s | Generation s | Sum after model load s |
|---|---:|---:|---:|---:|
| Shared-placement AR | 0.16 | 47.68 | 106.23 | 154.07 |
| Capacity AR | 0.17 | 26.98 | 116.03 | 143.18 |
| Fastest observed unpruned | 0.30 | 38.79 | 87.32 | 126.41 |
| Fastest observed pruned | 0.30 | 53.03 | 58.56 | 111.89 |

These are two-repeat measurements on one shared Orin/NVMe machine with short
raw prompts, single-request greedy decoding and one linear branch. Read-only
telemetry records temperature, CPU/GPU activity and device I/O; power, clocks,
other processes and system caches are not changed. The 8 GiB budget is a model/
runtime allocation guard, not a hard OS/cgroup limit. Throughput comparisons
must be read together with AR spread and the capacity controls.

```sh
python3 tests/benchmark-qwen35-ssd-speed.py --repo . \
  --model /path/to/Qwen3.5-35B-A3B-Q8_0.gguf \
  --output /path/to/new-results --frozen-math /path/to/frozen-math-ar.json
```

The harness requires every planned case to finish, validates exact token IDs,
GPU placement, direct-I/O payload, resident hits, phase accounting and memory
guards, and fails on model/binary/library changes. Failed evidence is retained.
The supplied frozen AR token array is checked after free generation and is not
used as generator input. A matching frozen reference is available in
`docs/validation/qwen35-cuda-ar-fix-20261018.json`, at
`suites.resident64.cases[0].tokens` (the original math AR control).
Collect kernel logs throughout the benchmark interval as well. The execution
harness's `status=passed` checks completion, tokens and accounting; it does not
certify storage health. Storage-fault assessment is a separate step, and this
published report explicitly uses `completed_with_storage_timeouts`.
