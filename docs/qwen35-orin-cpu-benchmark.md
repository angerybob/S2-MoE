# Orin CPU performance: experimental Qwen3.5 self-speculation

The audit-disabled CPU benchmark is slower than the same executable's AR mode
for every tested speculative configuration. AR averages **5.716 token/s**;
the fastest tested configuration (depth 1, expert top-k 4) averages **3.685
token/s**, or **0.645x AR throughput**. Correctness still passes in every run.
This is a result for the recorded CPU compatibility path, model and prompts;
it does not establish CUDA or SSD performance.

## Method and provenance

- Host: Jetson AGX Orin developer kit (`orin-60`), 12 CPU cores, approximately
  61 GiB RAM. MAXN power mode and `schedutil` governor were observed and left
  unchanged. GPU utilization recorded by `tegrastats` stayed at 0%; CPU
  temperature samples ranged from 50.937 to 59.593 degrees C.
- Model: original `ggml-org/Qwen3.5-35B-A3B-GGUF`, revision
  `3127ef0b7fd4f12626506419ceefde54fa8db53c`, Q8_0,
  SHA-256 `8a83fbfde74b0366feb76b075c10df46bbece6aa21ed14a6f291a7904f2b7d67`.
  The prior completed integrity check supplies this identity; model size/mtime
  and binary identity remain fixed throughout each comparison.
- Clean source revision: `3d69a2af8f5c997b8be49012318a23f9631a4f45`.
  Release CPU build, native ARM optimizations, OpenMP, CPU placement (`-ngl 0`),
  8 threads, context 128, ubatch 32, greedy/raw text, EOS ignored, 64 output
  tokens. Target uses all 8 routed experts; only Draft top-k changes.
- State-copy auditing is **off**. Token rendering is disabled with `--quiet`.
  No SSD loading, CUDA graphs or graph reuse are enabled in either mode.
- Main prompt: `Explain why the sky looks blue in two short sentences.`
  A full-length AR warmup is excluded. The 9 depth/top-k combinations each run
  twice in a fixed shuffled order (seed 35018), with an AR reference before
  each sweep and after the final sweep. Measured AR therefore has 3 samples.
- Throughput is **63 / post-prefill generation seconds**: the first of the 64
  tokens comes from prefill. Loading/context setup and prompt prefill are
  timed separately and excluded. Group rates use total decode tokens divided
  by total time, rather than an arithmetic mean of per-run rates.
- All **22** main-matrix runs, including warmup and AR references, emit exactly
  the same 64 token IDs. There are 18 measured speculative samples. This is a
  small workload comparison, without a confidence interval or universal claim.

[Machine-readable measurements and summary](validation/qwen35-orin-cpu-benchmark-20261018.json)
include source/binary identities, per-run timings, token IDs, acceptance counts,
OS resource counters and sampled machine state.

## Main matrix

Phase times below are mean seconds per 64-token output over the two repetitions.
State API time sums snapshot capture, pre-verification restore and accepted-prefix
commit. Acceptance pools accepted/proposed draft tokens across repetitions.

| Draft depth | Draft experts | Decode token/s | Relative to AR | Draft (s) | Verify + history (s) | State APIs (s) | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|
| AR | 8 | 5.716 | 1.000 | — | — | — | — |
| 1 | 1 | 2.999 | 0.525 | 6.178 | 12.764 | 1.780 | 60.0% |
| 1 | 2 | 3.613 | 0.632 | 5.191 | 10.537 | 1.468 | 93.9% |
| 1 | 4 | 3.685 | 0.645 | 5.212 | 10.212 | 1.428 | 100.0% |
| 4 | 1 | 1.841 | 0.322 | 15.292 | 17.577 | 1.162 | 38.4% |
| 4 | 2 | 2.823 | 0.494 | 10.054 | 11.404 | 0.739 | 73.4% |
| 4 | 4 | 3.355 | 0.587 | 8.631 | 9.416 | 0.623 | 94.3% |
| 8 | 1 | 1.175 | 0.206 | 27.203 | 25.305 | 0.989 | 22.7% |
| 8 | 2 | 2.286 | 0.400 | 14.058 | 12.870 | 0.537 | 58.4% |
| 8 | 4 | 3.330 | 0.583 | 9.669 | 8.774 | 0.398 | 93.2% |

AR's mean post-prefill generation wall time is **11.022 s**, including
**11.003 s** of target decode. The fastest speculative configuration takes
**17.096 s**; its individual sample rates are **3.641 / 3.731 token/s**.
The AR sample rates are **5.695 / 5.728 / 5.725 token/s**.

## What the phase measurements show

1. **Draft is only slightly cheaper per token than AR.** Routed top-k 1/2/4
   costs approximately 154-155 / 157-158 / 163-164 ms per proposed token,
   including greedy selection, compared with approximately **174.7 ms** per
   AR target decode. All recurrent, attention and shared-expert computation
   remains active. Measured expert sparsification alone produces a relatively
   small total Draft-cost reduction on this path.
2. **Verification does not amortize enough work to pay for Draft.** Verification
   with history averages approximately 160 / 141 / 127-129 ms per input token
   at draft depths 1 / 4 / 8. Increasing batch depth helps, but total Draft plus
   verification still exceeds AR. Depth 1/top-k 4 accepts **100%** of proposals
   and nevertheless runs slower. Depth 8/top-k 1 accepts only **22.7%**, so
   rejected proposals add substantial extra Draft and verification work.
3. **The separately measured state APIs are a secondary cost.** They account
   for approximately **1.8-8.5%** of speculative generation time. For the fastest
   configuration, snapshot / restore / commit cost **0.976 / 0.219 / 0.233 s**,
   totaling **1.428 s (8.35%)**. Even an ideal removal of those three costs would
   yield only approximately **4.02 token/s**, still below this AR reference.
   This arithmetic excludes any change to verification itself.

The code supplies a structural explanation worth profiling further:
[`build_delta_net_chunking`](../src/qwen3next.inc#L136) constructs dependent
single-token recurrent steps in a loop, including intermediate-state saves.
It is a compatibility implementation. The phase benchmark does **not** measure
individual operator/kernel shares or prove how much time a fused implementation
would save.

## Timing boundaries

`--timings-file` uses a steady wall clock and explicit backend completion at
phase boundaries. `draft_ms` includes draft decode and greedy selection;
`verify_with_history_ms` includes begin/decode/finish and the **in-graph writes
of prefix history**. Those history writes have not been isolated as a separate
kernel measurement. `snapshot_ms`, `restore_ms` and `commit_ms` measure whole
state API calls, including allocation/serialization, synchronization and KV-tail
bookkeeping where applicable; they are not pure memcpy bandwidth measurements.
Target greedy selection is reported separately as `sampling_ms`. Small remaining
loop/destruction costs remain in total generation time.

## Additional prompts

The fastest main-matrix configuration (depth 1/top-k 4) was then selected for
confirmation on two additional raw prompts. Each prompt has one measured
speculative sample, two measured AR samples and an excluded AR warmup, with
the same placement/settings and 64-token output length. All eight runs pass
per-prompt token equality. This is a small confirmation, not a second full grid.

| Prompt | AR token/s | Self-spec token/s | Relative to AR | Draft acceptance |
|---|---:|---:|---:|---:|
| [code](validation/qwen35-orin-cpu-confirm-code-20261018.json) | 5.711 | 3.709 | 0.649 | 100.0% |
| [math](validation/qwen35-orin-cpu-confirm-math-20261018.json) | 5.698 | 3.714 | 0.652 | 96.9% |


## Reproduce

Use a fresh output directory on a suitable Linux CPU host with the recorded
model and build. The opt-in harness never downloads weights or changes global
power/cache settings. Its child-only address-space limit defaults to 46 GiB;
adjust it explicitly for a different host.

```sh
python3 tests/benchmark-qwen-hybrid.py \
  --binary build-qwen35/bin/llama-speculative-qwen35 \
  --model /path/to/Qwen3.5-35B-A3B-Q8_0.gguf \
  --output /path/to/fresh-results \
  --tokens 64 --threads 8 --context 128 --ubatch 32 \
  --depths 1 4 8 --experts 1 2 4 --repeats 2
```

For a single invocation, add `--timings-file timing.json --quiet` to the
experimental executable. Leave `--audit-state-copies` off for this comparison.
The older audit-enabled 32-token correctness matrix remains a separate result.

The profiling change passed **19/19** Orin CPU regressions and **7/7** tiny-model
profiling/token-accounting runs. The [push CI](https://github.com/angerybob/S2-MoE/actions/runs/39096004930)
and [PR CI](https://github.com/angerybob/S2-MoE/actions/runs/39096007617) both passed
on the timed implementation revision. This does not claim all inherited,
repository-wide platform/type-check jobs pass.
