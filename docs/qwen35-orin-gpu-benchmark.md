# Historical Orin CUDA performance before the AR-consistency fix

The failure and timings below describe the pre-fix implementation. The math
failure is addressed in the [CUDA AR-consistency fix and SSD validation](qwen35-cuda-ar-fix.md).
The original raw failed report and measurements are retained; these throughput
numbers are not a post-fix performance benchmark.

**At this historical revision, the passed main GPU sweep is slower than AR
and the extra mathematics confirmation fails exact AR token equality.** This report promotes actual GPU measurements;
the earlier [CPU report](qwen35-orin-cpu-benchmark.md) is supplementary.

## Hardware, placement and method

- Jetson AGX Orin (`orin-60`), approximately 61 GiB unified RAM, MAXN power mode
  and `schedutil` CPU governor, both left unchanged.
- Original `ggml-org/Qwen3.5-35B-A3B-GGUF` Q8_0, revision
  `3127ef0b7fd4f12626506419ceefde54fa8db53c`, SHA-256
  `8a83fbfde74b0366feb76b075c10df46bbece6aa21ed14a6f291a7904f2b7d67`.
- CUDA Release build; **41/41 layers offloaded**, CUDA model buffer
  **34,667.78 MiB**, CUDA recurrent-state buffer **62.81 MiB**. Recorded GPU
  utilization reaches **99%**. This is not a CPU-placement benchmark.
- Clean timed source: `55a46245457b218c3f6cd3246fe51a9f18e63895`.
  Model is loaded once and remains resident. Each case creates fresh target,
  draft, KV and recurrent state; no previous case's live inference state is reused.
- `-ngl 99`, 8 CPU threads, context 128, ubatch 32, 64 generated tokens, greedy,
  raw prompts, EOS ignored, default F16 KV. Copy auditing and token rendering
  are **off**. AR is `--ar` in the same experimental executable. CUDA graphs and graph reuse
  are disabled on both paths by that tool.
  CUDA operator fusion retains its existing default configuration.
- Throughput is **63 / post-prefill generation seconds**. The first output token
  comes from prefill; loading, context setup and prefill are excluded and recorded
  separately. Stage timings explicitly wait for backend completion.
- Main prompt: `Explain why the sky looks blue in two short sentences.` All nine
  depth/top-k combinations run twice in seeded shuffled order. One full-length
  AR warmup is excluded; three AR measurements bracket the sweeps. Rates use
  total decode tokens / total time. Target always activates all 8 routed experts.

[Measurements, token IDs, placement evidence and coverage](validation/qwen35-orin-gpu-benchmark-20261018.json)
retain the original overall **failed** status from the confirmation failure.
The passed main subset is identified separately; there is no overall-pass claim.

## Passed main performance matrix

All **22 main runs** emit exactly the same 64 token IDs. There are 18 measured
self-spec samples and 3 measured AR samples, plus the excluded warmup.

AR averages **14.864 token/s**, or **4.239 s**
for the 63 post-prefill decode steps. The fastest tested speculative configuration
is depth 8/top-k 4 at **8.597 token/s**, **0.578x AR**,
taking **7.328 s**. All nine are slower than AR in this workload.
The top three configurations are close; this small sample does not establish a
universal optimal depth. Every phase below is mean seconds per 64-token output.

| Draft depth | Draft experts | Decode token/s | Relative to AR | Draft (s) | Verify + history (s) | Snapshot (s) | Restore (s) | Commit (s) | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.606 | 0.512 | 2.004 | 4.386 | 1.185 | 0.357 | 0.086 | 73.0% |
| 1 | 2 | 8.216 | 0.553 | 1.910 | 4.032 | 1.081 | 0.327 | 0.078 | 88.2% |
| 1 | 4 | 8.589 | 0.578 | 1.921 | 3.784 | 1.018 | 0.307 | 0.075 | 100.0% |
| 4 | 1 | 4.836 | 0.325 | 5.340 | 6.377 | 0.811 | 0.250 | 0.061 | 38.4% |
| 4 | 2 | 7.654 | 0.515 | 3.412 | 3.939 | 0.542 | 0.154 | 0.038 | 78.7% |
| 4 | 4 | 8.581 | 0.577 | 3.182 | 3.433 | 0.446 | 0.134 | 0.033 | 94.3% |
| 8 | 1 | 3.250 | 0.219 | 9.005 | 9.203 | 0.731 | 0.220 | 0.053 | 24.6% |
| 8 | 2 | 6.583 | 0.443 | 4.525 | 4.468 | 0.352 | 0.106 | 0.026 | 65.4% |
| 8 | 4 | 8.597 | 0.578 | 3.524 | 3.328 | 0.286 | 0.086 | 0.021 | 93.2% |

The main sweep observes 68 zero-accept,
82 partial-accept and 254 full-accept
rounds. Auditing is off during performance runs; these counters do not represent
an additional byte-for-byte state-copy audit.

## Where the time goes

For depth 8/top-k 4, Draft takes **3.524 s (48.1%)** and verification
including history writes takes **3.328 s (45.4%)**.
Snapshot / restore / commit take **0.286 / 0.086 / 0.021 s**, totaling
**0.394 s (5.37%)**. Sampling and other loop/destruction costs make
up the remaining approximately 1.1%.

- Draft is not sufficiently cheap: approximately **50-60 ms per proposed token**
  across these settings, compared with approximately **67 ms per AR decode**.
  Expert pruning is active, but all recurrent, full-attention and shared-expert
  paths remain active.
- Verification at the best configuration costs approximately **48.9 ms per input
  token**. It amortizes some work, but not enough to pay for the extra Draft pass.
  The depth 1/top-k 4 main case accepts 100% of proposals and still runs slower.
- The external state APIs are a secondary cost in the best configuration. Even
  removing all three ideally would yield only approximately **9.09 token/s**,
  below this AR reference. At depth 1 their share rises to approximately 19%,
  so their relative cost depends on verification-round count.

`verify_with_history_ms` **includes in-graph history writes**; they have not been
isolated as individual kernels. State API times include allocation/serialization,
synchronization and KV-tail bookkeeping, not only memcpy. No individual operator
or kernel time attribution is claimed. The recurrent compatibility graph still
expands dependent per-token steps, and the numerical-compatibility projection
policy described below constrains the available batch kernels.

## Additional prompts and the remaining blocker

The code prompt was checked at depths 1/8, top-k 4, one speculative sample each,
with two measured AR samples and an excluded warmup. All **5 code runs** pass
64-token equality; its AR average is **14.796 token/s**.

| Draft depth | Draft experts | Code token/s | Relative to code AR | Equality |
|---|---:|---:|---:|---|
| 1 | 4 | 8.569 | 0.579 | pass |
| 8 | 4 | 9.270 | 0.627 | pass |

The mathematics prompt is `Solve step by step: A shop sells 12 notebooks at
3 dollars each. What is the total price?` Its warmup/AR agree, but depth 1/top-k 4
first diverges at **output token 18** (zero-based index 17): AR token **2742**,
self-spec token **12**. That rate is excluded from valid performance aggregates.
The harness stopped, leaving the planned math depth 8/top-k 4 and final AR
unexecuted in this performance suite: **30 of 32 cases completed, 29 valid**.

A separate diagnostic reproduces the math mismatch at the same position with
depth 1/top-k 2, depth 1/top-k 4 and depth 8/top-k 4. Repeated fresh AR contexts
before, between and after those cases remain identical. From an identical saved
AR prefix, a first-row single-token/two-token forward probe without history capture gives identical
logits (`max_abs = 0`); that probe alone does not explain the accumulated
self-spec divergence. The exact remaining multi-token/history-path cause is
**unresolved**. Diagnostic traces are excluded from throughput results.

This limitation blocks general GPU AR-equivalence qualification. Keep this
contribution experimental and the PR in Draft; do not infer complete GPU support
from the passed main matrix or the earlier small fixtures. The existing CPU qualification does not establish GPU correctness. Other
quantized layouts, long contexts, vision/MTP, sampling, multi-request/tree
speculation and SSD/hot-expert integration remain distinct scopes.

## Q8 consistency fix uncovered by GPU testing

The [initial GPU sweep](validation/qwen35-orin-gpu-initial-failed-20261018.json)
failed on the main prompt at output token 48. A controlled same-prefix probe
showed a maximum logit difference of **0.176672** before any history commit.
Quantized CUDA expert projections selected MMVQ for serial AR and MMQ for a
batch, using different activation arithmetic; dense Q8 projections had the
same issue for larger batches.

Commit `8b2a294d7` adds an opt-in compatibility policy, automatically enabled by
the experimental hybrid executables as `GGML_CUDA_Q8_AR_COMPAT=1`. For explicit
F32 Q8 projections it keeps the serial vector reduction geometry while mapping
tokens to independent GPU channels/sample blocks. Strided top-k IDs are packed
when needed. Ordinary CUDA callers without that opt-in retain existing dispatch.

The new CUDA regression checks **20** dense/expert cases at batch sizes
2/5/9/32, including contiguous and strided expert IDs: every result is
**bit-identical** to the per-token oracle after the fix. Before it, projection
errors reached approximately **0.00368**. CPU regressions remain **19/19**;
resident CUDA fixture checks cover fresh contexts across multiple prompts.
This fixes the reproduced main-prompt issue, **not** the remaining math failure.

## Reproduce

```sh
python3 tests/benchmark-qwen-hybrid.py \
  --binary build-next/bin/llama-speculative-qwen35 \
  --model /path/to/Qwen3.5-35B-A3B-Q8_0.gguf \
  --output /path/to/fresh-gpu-results \
  --gpu-layers 99 --resident-model --confirm-prompts \
  --tokens 64 --threads 8 --context 128 --ubatch 32 \
  --depths 1 4 8 --experts 1 2 4 --repeats 2
```

On the recorded revision, `--confirm-prompts` intentionally surfaces the known
math failure and exits nonzero. Omit that flag to reproduce only the passed main
sweep; the limitation remains documented. Leave the virtual-address-space cap
at 0 for CUDA. The resident child has a 2.5 GiB physical `MemAvailable` guard and
only its own process group is stopped on failure. No power/governor/cache
settings are modified.

To run the small CUDA regression:

```sh
cmake --build build-next --target test-cuda-moe-batch
ctest --test-dir build-next -R qwen35-cuda-moe-batch --output-on-failure
```

The timed revision's [Qwen3.5 CI](https://github.com/angerybob/S2-MoE/actions/runs/39100078964)
and [controlled type-check baseline](https://github.com/angerybob/S2-MoE/actions/runs/39100078958)
passed. This does not claim all inherited repository-wide checks are green.
