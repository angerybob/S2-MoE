# Experimental Qwen3.5-MoE integration

Development branch: `feature/qwen35-hybrid-self-spec`, based on merged PR #2.
This is a separate integration; Qwen3-Next validation does not qualify Qwen3.5.

Current GPU numerical qualification is recorded in the
[CUDA AR-consistency fix and completed SSD comparisons](qwen35-cuda-ar-fix.md).
It covers four prompts at 64 tokens plus 8 GiB 32-token and pruned-tree checks;
the older architecture/performance records below retain their original scope.

The subsequent [hybrid S2-MoE integration](qwen-hybrid-s2moe.md) connects this
architecture to budgeted GPU SSD loading, resident experts, adaptive draft
trees/pruning, optional verification-time reuse and the main executable.
The earlier correctness and performance evidence below remains scoped to its
recorded configurations; it does not qualify new deployment modes automatically.

Acceptance is staged:

1. Register `qwen35moe`, implement the distinct `qwen35` pre-tokenizer, load the
   separate alpha/beta projections and tiled GDN key heads, and establish a
   reproducible text-only autoregressive reference. The new
   `llama-speculative-qwen35` entry point provides `--ar` as its ordinary
   greedy reference mode.
2. Enable expert-only drafting after AR validation. Preserve every recurrent,
   full-attention and shared-expert path; audit routed-expert counts and ensure
   the target keeps its original expert count.
3. Enable snapshots and verification-prefix history after draft validation.
   Recheck root-only/zero acceptance, partial/full acceptance, state restoration,
   correction continuation and AR equivalence for Qwen3.5 separately.

The compatibility graph reuses Next's recurrence implementation with explicit
architecture-specific alpha/beta projections and key-head mapping. Text uses
ordinary scalar positions and the NEOX reduction of canonical interleaved
IMRoPE sections. Layouts with uncovered rotary pairs are rejected. Vision,
multimodal positions, MTP,
CUDA graphs, graph reuse, sampling, and multi-request speculation are not
supported by this experimental entry point. No performance claim is made.

Source references: upstream llama.cpp `src/models/qwen35moe.cpp` and
`src/llama-vocab.cpp` (MIT; https://github.com/ggml-org/llama.cpp).
Validation reports will identify the exact reference revision and model hash.

## Initial validation status

- CPU builds of the Qwen3.5 AR entry point and the independent AR driver passed
  on Jetson Orin.
- The real Qwen3.5 vocabulary matches upstream token IDs in 12 multilingual,
  combining-mark, digit, whitespace, emoji and special-token cases. This uses
  a vocabulary-only GGUF extracted from the original metadata, not a trained
  model forward pass.
- A 40-token F32 random-weight fixture case matches upstream greedy choices.
  The independent three-prompt regression matrix is run by `qwen35.yml`.
- Existing Next fixture/state/greedy tests passed 9/9 after graph integration.
- Real-model Q8_0 CPU AR is now validated against the pinned upstream reference.
  Expert-only drafting audits pass on CPU and CUDA fixtures and the real Q8_0
  model. Verification-history, self-spec fixture checks and the recorded
  real-model 32-token self-spec matrix now pass.

Pinned reference source: upstream llama.cpp
`650bf14eb9a922de0f88c9523a271159cc5ae469`.
Selected real-model artifact: `ggml-org/Qwen3.5-35B-A3B-GGUF`, revision
`3127ef0b7fd4f12626506419ceefde54fa8db53c`, file
`Qwen3.5-35B-A3B-Q8_0.gguf`, SHA-256
`8a83fbfde74b0366feb76b075c10df46bbece6aa21ed14a6f291a7904f2b7d67`.

To build the candidate driver, build target `test-qwen35-ar`. Compile the same
`tests/qwen35-reference-ar.cpp` unchanged against the pinned upstream headers and
CPU libraries, then run `tests/validate-qwen35-ar.py --help` for the comparison
harness. Both drivers use greedy sampling, ignore EOS, disable attention flash
kernels and use CPU-only model placement. The candidate also disables graph reuse.

### Recorded Gate 1 checks

| Check | Result | Scope |
|---|---|---|
| Independent upstream vs candidate, 3 prompts x 32 tokens, batch/serial prefill | 6/6 exact token sequences | F32 random-weight fixture, Orin CPU |
| Same independent comparison on x86 CPU | 6/6 | 2x-A100 host, CPU-only inference |
| CPU vs CUDA AR entry point, 3 prompts x 32 tokens | 3/3 exact token sequences | F32 random-weight fixture, Orin |
| Original real-model vocabulary vs pinned upstream | 12/12 exact token sequences | Vocabulary-only metadata; no weight inference |
| Existing Next regression | 9/9 | F32/F16/BF16 fixtures, state/history, greedy loop |

Machine-readable reports: [independent AR fixture](validation/qwen35-ar-fixture-20261017.json),
[real tokenizer](validation/qwen35-tokenizer-20261017.json),
[Orin CUDA AR](validation/qwen35-cuda-ar-fixture-20261017.json).
These are not full real-model Qwen3.5 forward-pass or self-spec validation.

An existing full HF Qwen3.5-35B-A3B snapshot on `2x-A100` was converted separately
with the pinned upstream converter to Q8_0. This local conversion is distinct
from the ggml-org download above. Its SHA-256 is
`f23b26ea23f30a491728b1e4448f22e1bcbe69805bba8618f2277f18ac3cdaea`.
The [conversion manifest](validation/qwen35-model-conversion-20261017.json)
records the source shard sizes/mtimes, metadata hashes and converter revision.

### Real-model Gate 1 and expert-only Gate 2

The real-model [AR report](validation/qwen35-real-ar-20261017.json) records three
raw prompts, 32 generated tokens each, and candidate batch/serial prefill against
upstream: **6/6 exact prompt/output token sequences** on the `2x-A100` CPU host.
The actual `llama-speculative-qwen35 --ar` entry point also reproduced the first
32-token upstream reference with default F16 KV and eight CPU threads.

The [expert audit](validation/qwen35-expert-audit-20261017.json) inspects executed
up/gate/down projection shapes. On the real model, top-k 1/2/8 retain all 30
recurrent layers, 10 full-attention layers and 40 shared-expert paths. Routed
projection output elements are 362496 / 724992 / 2899968 respectively. Setting
draft top-k to 1 on a target context leaves its logits exactly unchanged and its
routed expert count at 8. These graph-size reductions do not establish speedup.
The same audit passes on Orin CPU and CUDA F32 fixtures (top-k 1/2).

### Gate 3: hybrid memory integration

The experimental entry point now supports expert-only self-speculation as well as
`--ar`. Both contexts share hybrid memory; drafting is undone using a recurrent
snapshot, and verification history commits the target-computed accepted-prefix
state directly, including root-only acceptance and correction continuation.

- Orin CPU: **19/19** combined Next/Qwen3.5 tests passed. The 10 Qwen3.5 tests
  include fixture generation, state/history and end-to-end greedy equality across
  F32/F16/BF16 weights, plus the expert-only audit.
- Independent x86 CPU build: **19/19** combined tests passed on `2x-A100`.
- Orin explicit CUDA: **6/6** Qwen3.5 state/history and greedy-loop checks passed
  across F32/F16/BF16 weights. State/logit tests use F32 KV; executable tests use
  the default F16 KV.
- Prefix tests cover every accepted prefix for verification lengths 1, 2, 4, 9
  and 32, subsequent correction tokens, 64-token prefill, capacity/lifecycle
  misuse, multiple sequences and split verification batches.

The real Q8_0 CPU self-spec matrix passed as recorded below. SSD/hot-expert
loading is now connected through the hybrid S2-MoE driver and has separate
validation. Quantized formats other
than the recorded Q8_0 model, fused recurrent projections, partial GPU offload,
CUDA configurations beyond the recorded GPU comparisons, MTP, vision,
stochastic sampling and multi-request speculation remain unqualified. Bounded DFS draft trees use staged path
verification, as described in the current deployment guide.

Example (CPU, one greedy chain):

```sh
build/bin/llama-speculative-qwen35 -m model.gguf -ngl 0 -p "Explain why the sky looks blue." -n 32 --draft-max 8 --draft-expert-topk 2 --audit-state-copies
# Add --ar to obtain the same entry point's ordinary greedy reference.
```

Use `tests/validate-qwen3next-real.py` with the Qwen3.5 executable/model and
`--gpu-layers 0` for the opt-in real-model matrix. The harness is shared with Next;
it records actual model/binary identities and the verification commit audits.

### Extended real-model self-spec validation

Inference revision: `d7e1ce53aaeb540435d52030d56d7ab398315723`. The model and binary
were fixed throughout the [real-model matrix](validation/qwen35-real-selfspec-20261017.json).
Three raw prompts generate 32 tokens each, with draft depth/top-k `(1,1)`, `(4,2)`
and `(8,4)`, greedy decoding, EOS ignored for fixed-length comparisons, and CPU
placement on `2x-A100`:

- **9/9** speculative sequences exactly match their AR reference token IDs.
- The three product AR sequences also match the independently built upstream
  reference from Gate 1; this comparison uses the recorded token IDs.
- **21 zero-accept, 8 partial-accept and 65 full-accept rounds** were audited.
- All native verification-prefix copy audits and accounting checks pass;
  accepted-prefix replay count is **0**.

| Prompt | Draft depth | Expert top-k | Accepted/proposed | Verification rounds |
|---|---:|---:|---:|---:|
| science | 1 | 1 | 12/20 | 20 |
| science | 4 | 2 | 23/34 | 9 |
| science | 8 | 4 | 28/28 | 4 |
| code | 1 | 1 | 12/19 | 19 |
| code | 4 | 2 | 24/30 | 8 |
| code | 8 | 4 | 28/28 | 4 |
| math | 1 | 1 | 14/18 | 18 |
| math | 4 | 2 | 25/28 | 7 |
| math | 8 | 4 | 27/37 | 5 |

The measured history-buffer allocations are 125.625 MiB (capacity 2),
314.0625 MiB (capacity 5), and 565.3125 MiB (capacity 9). Each accepted-prefix
commit copies 65,863,680 bytes (62.8125 MiB) of recurrent state. Auditing reads
back additional bytes; snapshots and compatibility-graph buffers are separate
costs. These are memory/copy measurements, not a throughput benchmark.

A separate [Orin CPU real-model smoke test](validation/qwen35-orin-real-smoke-20261018.json)
uses the original ggml-org Q8_0 artifact after verifying its published size and
SHA-256. With 8 output tokens, depth 4/top-k 2, AR and self-spec token IDs match;
zero/full acceptance and native copy audits pass. This short Orin run does not
replace the longer 32-token matrix on the x86 CPU host. GPU qualification is
recorded separately in the [current CUDA/SSD report](qwen35-cuda-ar-fix.md).

The [Qwen3.5 CPU workflow](https://github.com/angerybob/S2-MoE/actions/runs/39051393156)
passed on the implementation revision. Later report/documentation changes do
not change inference behavior. This experimental contribution is related to
issue #1 and does not close it or claim universal equivalence or acceleration.

### Orin GPU performance and qualification follow-up

The historical pre-fix Q8_0 CUDA sweep offloaded all 41 layers. Its main
64-token matrix passed all 22 runs but showed no speedup, while the extra math
prompt exposed the numerical blocker. See the
[preserved GPU timing/failure record](qwen35-orin-gpu-benchmark.md). The later
[AR-consistency fix](qwen35-cuda-ar-fix.md) resolves that recorded failure and
provides separate GPU SSD correctness checks; it does not relabel the old timing
run as a current benchmark. The CPU report below remains supplementary.

### Supplementary Orin CPU performance follow-up

A separate 64-token benchmark now compares AR with depth 1/4/8 and expert
top-k 1/2/4, with auditing and token rendering disabled. On this recorded CPU
path, all nine speculative configurations are slower than AR: 5.716 token/s
for AR versus 3.685 token/s for the fastest tested configuration. See the
[complete matrix, phase timings and reproduction instructions](qwen35-orin-cpu-benchmark.md).
These CPU measurements do not establish full-model CUDA or SSD performance.
