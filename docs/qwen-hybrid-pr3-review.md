# PR #3 review and CI attribution

This experimental contribution integrates Qwen3-Next and Qwen3.5-MoE with the
S2-MoE entry point, expert placement, expert-only drafting, bounded hybrid-state
history and pruning. The base reviewed here is `orin` at
`bc4129bd23943a8e6542304b03ce18f74cf04549`. The implementation remains limited to
one request, sequence 0, raw text and greedy decoding. CUDA graphs/reuse are
disabled. Approximate expert reuse is separately opt-in and is not included in
the exact-equivalence or speedup evidence.

## CI failures

The [controlled CI inspection](https://github.com/angerybob/S2-MoE/actions/runs/39257175549)
compares both revisions in one runner with the same dependencies. The
[machine-readable record](validation/qwen-hybrid-pr3-review-20261019.json)
contains individual failed-job links and source checks.

| Check | Evidence and disposition |
|---|---|
| New Python lint failures | Five `NP100` diagnostics in four new validation/benchmark scripts were introduced here. Fixed in `e87c4921d8` by using logging for progress messages. Before: baseline 50 / candidate 55. After: 50 / 50, zero added or removed diagnostic identities. Existing lint policy is unchanged. |
| Fatal-warning CPU builds | Both revisions were configured and built with fatal warnings enabled. Both fail in `ggml/src/ggml-cpu/ops.cpp` because the array `tensor::name` is tested as a nullable pointer; the captured error identity is identical. This comparison stops there and does not establish a clean all-target build. Other observed unused-variable, declaration and Qwen3VL-switch error sites are also present in the base source. |
| Windows checkout | The base already tracks a root filename consisting of U+0001. The identical tree entry remains in this PR; Windows jobs fail during checkout with `invalid path '?'`, before compilation. This is not repaired by changing this feature's build scripts. |
| HIP/MUSA | Observed failures reference CUDA-only allocation/copy/pointer APIs in existing SSD backend code. All inspected error-site source lines are present in `orin`. These backends are not newly qualified; this contribution's SSD execution requires CUDA. |
| Python types | The separate controlled baseline workflow retains the same dependencies and checks both revisions. Its earlier record has 17 errors and one warning on each revision, with no new diagnostics; the [follow-up on the lint fix](https://github.com/angerybob/S2-MoE/actions/runs/39257183856) also passes. |
| Other jobs | Queued, cancelled or uninspected platform jobs are not represented as passing. Repository-wide CI is not all green. |

The logging changes preserve the non-logging Python AST in all four scripts.
The executed speed harness is linked by immutable commit and SHA-256 in the
[speed report](qwen35-ssd-postfix-speed.md); historical measurements were not
regenerated or relabelled.

## Code self-review

- **State transitions:** drafting starts from a captured target prefix. The
  shared memory is restored before target verification. The committed history
  prefix includes the root even when no draft token is accepted. Single-token
  verification retains its live state, rather than copying an uninitialized
  history row. Commit invalidates the consumed history and removes the rejected
  attention tail; the correction token is the next round's root.
- **Branches:** DFS snapshots are restored only to ancestors whose attention
  prefix has not been overwritten. Verification follows contiguous paths;
  switching to an alternative commits the target prefix before continuing.
  This is staged path verification, not arbitrary best-first state restoration.
- **CUDA arithmetic:** compatibility hints are scoped to hybrid target graphs.
  Scalar AR and all-token verification retain the same reduction geometry;
  expert ID packing handles strided top-k views. The float and Q8 operator tests
  cover ordinary/strided layouts and resident/SSD execution.
- **Placement and budget:** packed expert aliases preserve shard indices,
  quantization, lengths and offsets. Non-routed weights stay resident; placement
  selects whole expert bundles within the remaining weight budget. Oversized
  hot lists and insufficient runtime reserves fail explicitly. Runtime
  accounting includes shared state once, compute buffers, host snapshots and
  SSD staging. This is buffer accounting, not an OS process-RSS guarantee.
- **I/O and unsupported settings:** interrupted reads are retried, aligned
  direct reads retain their leading offset, and missing/short expert reads fail
  instead of silently computing with absent data. CLI/config checks reject
  unsupported sampling, memory layouts, history sizes and external draft models.
  Vision/MTP and general multi-request decoding are outside this release.

No blocking issue was found in these reviewed paths. On 2026-10-19 the Orin
CUDA build reran **35/35** tests successfully in 87.84 seconds, including actual
GPU float/Q8 tests and both architectures' GPU/SSD integration tests. The native
inference files are unchanged from the previously qualified fix. The broader
[real-model correctness matrix](qwen35-cuda-ar-fix.md) remains the evidence for
longer Qwen3.5 GPU output, including memory-limited SSD placement.

The [Next workflow](https://github.com/angerybob/S2-MoE/actions/runs/39257183881)
and [Qwen3.5 independent-reference workflow](https://github.com/angerybob/S2-MoE/actions/runs/39257183927)
also pass on `e87c4921d8`. The final follow-up adds this review record and removes
the completed inspection helper; it changes no inference or test execution.

The completed post-fix speed experiment has 28/28 exact speculative matches,
but 181 device timeout/error events contaminate throughput. Neither these
timings nor this review establish stable or general speedup. These limitations
remain part of the experimental release; issue #1 remains open for follow-up.
