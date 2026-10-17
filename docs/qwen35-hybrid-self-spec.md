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
- Full real-model Qwen3.5 AR, expert drafting and verification-history acceptance
  are **not yet qualified**. The real model is still being downloaded and will
  be hash-verified before inference.

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
