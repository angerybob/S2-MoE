<!-- qwen3next-progress-pr2-20261017 -->
Hi @haochen76,

A quick update on our progress: we have published our experimental Qwen3-Next expert-only self-speculative decoding implementation in Draft PR #2:
https://github.com/angerybob/S2-MoE/pull/2

It includes shared hybrid-memory state restoration and verification-prefix history, allowing target-computed recurrent states to be committed without replaying the accepted prefix.

Additional validation on a real Qwen3-Next-80B-A3B model passed all 9 prompt/configuration cases: each produced exactly the same 32-token greedy output as standard autoregressive decoding, covering zero, partial, and full draft acceptance.

This currently supports Qwen3-Next, single-request, single-branch greedy decoding only. It is not yet Qwen3.5-MoE support, and no speedup claim is being made. We plan to use this implementation as the foundation for Qwen3.5-MoE integration.

Thanks again for suggesting this extension!
