# Qwen3.5 CUDA AR-consistency fix and bounded-SSD validation

Subsequent [GPU/SSD timing measurements](qwen35-ssd-postfix-speed.md) use
these same corrected binaries. That speed matrix detects NVMe completion
timeouts and reports fault-affected timings separately from correctness.

The previously recorded Qwen3.5 GPU math failure at output index 17 is fixed
for the tested configurations. The new AR and self-spec sequences match the
**unchanged pre-fix ordinary-AR reference**, including all 64 tokens of the
original math prompt. The SSD matrix completes **19/19** cases: seven AR/control
runs and twelve exact speculative runs, totaling 1024 emitted tokens.

Source: `a7128ceb96d01349fdf895de6b8c4235049af2c0`. Model: original ggml-org
Qwen3.5-35B-A3B Q8_0, previously verified SHA-256
`8a83fbfde74b0366feb76b075c10df46bbece6aa21ed14a6f291a7904f2b7d67`.
The harness records model size/mtime and binary/library hashes and rejects
changes during a run. All inference uses Orin CUDA with 41/41 layers offloaded;
host and CUDA allocations share Jetson LPDDR.

## Diagnosis

Under the same forced AR prefix, two-token batched forward already differed
from two serial steps **without history capture**. At the initial prefix,
row zero matched but row one had a maximum logit difference of 0.964263558;
history capture produced the same difference. At another prefix the wrong
forward values also affected the committed state and subsequent logits.
Disabling CUDA operator fusion did not eliminate the discrepancy.
Early token agreement therefore concealed state divergence before the first
different emitted token.

The remaining floating matrix operations used different reduction geometry:
serial AR selected vector kernels while verification could select GEMM. F32
accumulation alone does not make the addition order identical. Standalone
F16/F32/BF16 matrix tests exposed differences of a few ulps; the trained Q8 model
amplified these small differences downstream. Quantization and routed-expert
selection are sensitive to such boundaries; this record does not claim a
bin-by-bin trace of every amplification step.

The fix marks scalar AR and all-token-logit target verification with a per-node
CUDA dispatch hint. Token columns become independent vector samples while
retaining grouped-head and tensor strides. Compatible F32/F16/BF16 operations
use the same reduction geometry; Q8 sorted expert operations propagate the
hint. The hint is independent of profiling and copy audits. Ordinary multi-token
last-logit-only prefill retains its prior dispatch.

Verification remains a batched target graph, with parallel CUDA samples and
grouped SSD expert loads. Accepted target prefixes are committed without replay.
This does not substitute a serial target replay for speculative verification.

The final same-prefix diagnostic checks offsets 0, 8 and 16: row-zero/row-one
logits, every recurrent R/S array, the selected history state, and continuation
all have **max_abs = 0** against serial AR. The utility uses forced tokens only
to isolate the computation. The end-to-end runs below freely generate tokens;
their frozen-reference option performs only an afterward comparison.

## Regression and the original failure

- Orin CUDA build: **35/35** CTest targets pass (31 host/state/CLI/metadata and
  four actual GPU targets).
- **192** floating matrix comparisons are bit-identical to serial results:
  F32/F16/BF16, inner sizes 64/128/256/1024, 64/1024 output rows, scalar/grouped
  heads, and batch widths 2/5/9/32. This includes shapes beyond the previous
  vector-kernel performance heuristic. The existing **36** Q8 resident/SSD
  comparisons also pass bit-identically.
- A fully resident diagnostic run completes **13/13** 64-token comparisons
  across math, science, code and Chinese, including a repeated math AR control.
  All math outputs equal the frozen pre-fix AR sequence; the historical bad
  speculative token 12 at index 17 is replaced by the AR token 2742. This run
  reproduces and isolates the former resident failure, rather than serving as
  the SSD-deployment qualification.

## GPU + SSD under memory budgets

All suites use `-ngl 99 -c 128 -ub 32 -t 8`, raw prompts, greedy decoding,
ignored EOS for fixed lengths, quiet output and copy auditing off. CUDA graphs
and graph reuse remain disabled. Approximate expert reuse is explicitly off.
Each suite loads weights once and creates fresh logical state for every case.

All non-routed weights remain resident. The 24 GiB / 3 GiB reserve suite retains
about 21 GiB of model buffers and 5948/10240 expert bundles; the 8 GiB / 3 GiB
reserve suites retain about 5 GiB and 808/10240 bundles. Remaining experts have
no model-buffer allocation and use direct SSD I/O. Every case checks positive
resident hits **and** cold SSD payload reads, plus the reported model/runtime
allocation budget. This is an allocation guard, not an OS/cgroup hard limit.

| Total / runtime reserve | Prompts | Output tokens per run | Speculative settings | Speculative AR matches | All cases |
|---|---:|---:|---|---:|---:|
| 24 GiB / 3 GiB | 4 | 64 | D1/K2, D8/K4 | 8/8 exact | 13 |
| 8 GiB / 3 GiB | 1 | 32 | D1/K2, D4/K2, D8/K4 | 3/3 exact | 4 |
| 8 GiB / 3 GiB | 1 | 32 | tree + pruning, D4/K2 | 1/1 exact | 2 |

The 64-token prompts cover the original notebook-price math question, sky-color
explanation, a Fibonacci coding request, and a Chinese moon-phase explanation.
The 24 GiB suite repeats math AR at the end. All math cases in both budgets,
including the pruned tree, also match the frozen pre-fix AR prefix. Exact token
IDs, zero/partial/full acceptance counts, histories, SSD counters and complete
commands are in the [machine-readable report](validation/qwen35-cuda-ar-fix-20261018.json).

Across these SSD runs, the recorded zero/partial/full acceptance counts are
14/5/186.
The real 32-token pruned-tree run prunes
50 expansions and records
1 alternate-child switch, while still matching AR.

## Reproduction

```sh
cmake --build build-next --target llama-speculative-qwen35 \
  diagnose-qwen-hybrid test-cuda-float-batch test-cuda-moe-batch -j4

python3 tests/validate-qwen35-gpu.py \
  --binary build-next/bin/llama-speculative-qwen35 --model /path/to/model.gguf \
  --output /path/to/new-24g-results --tokens 64 --memory-budget-mib 24576 \
  --runtime-reserve-mib 3072 --configs 1:2 8:4 --repeat-math-ar

python3 tests/validate-qwen35-gpu.py \
  --binary build-next/bin/llama-speculative-qwen35 --model /path/to/model.gguf \
  --output /path/to/new-8g-results --tokens 32 --memory-budget-mib 8192 \
  --runtime-reserve-mib 3072 --prompts math --configs 1:2 4:2 8:4

python3 tests/validate-qwen35-gpu.py \
  --binary build-next/bin/llama-speculative-qwen35 --model /path/to/model.gguf \
  --output /path/to/new-tree-results --tokens 32 --memory-budget-mib 8192 \
  --runtime-reserve-mib 3072 --prompts math --configs 4:2 --branches 2 --prune
```

`--expected-math path/to/frozen-ar-tokens.json` additionally verifies that the
ordinary AR reference has not changed. The harness fails and preserves evidence
on any token, residency, allocation, runtime-file or completion mismatch.

## Scope

These results resolve the recorded greedy GPU blocker for this checkpoint and
the tested 32–64-token configurations. They do not prove universal equivalence
for arbitrary prompts, model formats, contexts, devices or sampling modes.
Approximate expert reuse still requires separate quality evaluation. No speedup
claim is made from these correctness runs; historical pre-fix timing/failure
reports are retained as historical evidence rather than rewritten.
