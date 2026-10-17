# Experimental Qwen3.5-MoE integration

Development branch: `feature/qwen35-hybrid-self-spec`, based on merged PR #2.
This is a separate integration; Qwen3-Next validation does not qualify Qwen3.5.

Acceptance is staged:

1. Register `qwen35moe`, implement the distinct `qwen35` pre-tokenizer, load the
   separate alpha/beta projections and tiled GDN key heads, and establish a
   reproducible text-only autoregressive reference. The new
   `llama-speculative-qwen35` entry point currently **requires `--ar`**.
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
CUDA graphs, graph reuse, sampling, and multi-request/tree speculation are not
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
  model. Verification-history/self-spec acceptance is still pending.

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

Gate 3 remains closed: the product entry point still requires `--ar` until
snapshot/history and accepted-prefix continuation tests pass for Qwen3.5.
