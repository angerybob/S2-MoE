# Qwen3-Next pre-merge self-review (2026-10-17)

Scope: PR #2, base `orin` at `48e952cfc48862b943f409d5374802e4a771dbb5`.
The implementation and validation evidence snapshot is `75aee813d`.
Follow-up changes fix a lint continuation and explicitly include `<algorithm>` in
`llama-context.cpp` for portable standard-library algorithm declarations. They do
not change decoding arithmetic or acceptance semantics.

## Code review

Reviewed the loader and architecture registration, expert-only draft selection,
shared-memory ownership, root/correction-token handling, recurrent snapshot and
verification-prefix history lifecycle, graph copy ordering, single-token commit,
KV trimming, explicit F32 CPU/CUDA precision paths, fixture tests, real-model
harness, and unsupported-configuration rejection. No blocking correctness issue
was identified within the documented experimental scope. This is a self-review;
it is not an independent reviewer approval.

The real-model matrix remains 9/9 matching 32-token greedy comparisons, with 31
zero-, 14 partial-, and 57 full-accept rounds. This establishes the tested cases
only. Qwen3.5, graph reuse, stochastic decoding, and performance are not qualified.

## CI triage

- Next CPU correctness passed on the published head `110e8a3eb`; the final head
  must pass the same workflow before readiness/merge.
- Controlled Python type-check comparison: 17 errors and 1 warning on each
  revision, identical normalized diagnostics, zero added/removed. See the
  committed type-check report and run `39019436391`.
- Controlled lint comparison initially found 50 baseline diagnostics plus one
  added E128 in the new comparison script. The continuation has been rewritten;
  finalization checks baseline/candidate again in one environment and requires
  zero added diagnostics. Existing Python scripts are not changed.
- Windows jobs fail during checkout because the base already contains the root
  control-character filename `\001`; no model code is compiled in those jobs.
- ASan/fatal-warning builds report the existing `t->name` address check in
  `ggml/src/ggml-cpu/ops.cpp:25`. That file is identical to the base.
- Other fatal-warning builds report existing unused variables/missing declarations
  in `llama.cpp`/`llama-context.cpp` and the unhandled Qwen3-VL enum. Those reported
  statements also occur on the base. They are not repaired in this feature PR.
- WebGPU's full example build reports the pre-existing unconditional
  `nvtx3/nvToolsExt.h` include in `examples/speculative/speculative.cpp`.
- Android's SDK-manager dependency step fails before compilation.
- iOS builds the library successfully, then fails packaging because the workflow
  calls `build-xcframework.sh`, which is absent on the base as well.
- RISC-V reported missing standard algorithm declarations in `llama-context.cpp`;
  an explicit `<algorithm>` include is added to this touched file.

Evidence collection: Actions runs `39042007097`, `39042108334`, `39042338625`.
The broad inherited platform matrix is not all green and does not establish
cross-platform support. Experimental CPU/Orin correctness and the documented
baseline comparisons are the acceptance checks for this contribution.

Issue #1 remains open for Qwen3.5-MoE. Progress comment:
https://github.com/angerybob/S2-MoE/issues/1#issuecomment-6235250798
