# Experimental Qwen3-Next self-speculation

This contribution ports the maintainer's earlier Jetson Orin prototype onto the
public `orin` branch. It adds `qwen3next` GGUF loading and an explicit, limited
`llama-speculative-next` executable, which accepts Qwen3-Next models only.
Qwen3.5 is a separate experimental entry point; see
[Qwen3.5 integration and validation](qwen35-hybrid-self-spec.md) and
[issue #1](https://github.com/angerybob/S2-MoE/issues/1).
General hybrid-model support is not implied by either integration.

The subsequent [hybrid S2-MoE integration](qwen-hybrid-s2moe.md) adds budgeted
SSD placement from packed/split GGUFs, adaptive trees/pruning, optional expert
reuse and main-entry dispatch. The initial state mechanism and validation
history are retained below.

## Scope

- One request, sequence zero, one shared hybrid memory; linear drafts or bounded
  DFS draft trees with staged path verification.
- Target and draft share the same model weights. Only the draft's routed MoE
  expert count is reduced. Gated DeltaNet, full attention, and shared experts
  execute in every layer.
- Greedy sampling, raw prompts, a bounded output length, and no context shifting.
- Draft depth 0–31. The history capacity is `draft_max + 1`, including the known
  target root. Verification must fit in one microbatch (`-ub >= draft_max + 1`).
- CUDA graphs and llama graph reuse are disabled automatically. Layer skipping,
  multiple requests and stochastic acceptance remain unsupported. Unsupported CLI
  options fail rather than silently enabling an unsafe combination.
- Dense and indirect expert matrix operations request F32 accumulation. The CUDA
  backend preserves this request through expert sorting and cuBLAS fallbacks,
  avoiding TF32/half-input reductions that differ between batched and serial
  execution. This conservative precision choice has a performance cost; this
  initial contribution makes no throughput or speedup claim.

The `llama-speculative` executable now dispatches Next models to this hybrid
driver; the original Transformer tree decoder cannot rewind recurrent state.
Existing Transformer paths retain their behavior.

## Build and run

On a configured Jetson Orin CUDA toolchain:

```bash
cmake -S . -B build-next -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-next --target llama-speculative-next test-qwen3next-state -j4

build-next/bin/llama-speculative-next \
  -m /path/to/qwen3-next.gguf -p 'The capital of France is' \
  -ngl 99 -c 512 -ub 32 -n 32 --draft-max 8 --draft-expert-topk 2
```

Use an existing Qwen3-Next GGUF with packed expert tensors for a resident run.
The loader also accepts the prototype's per-expert split layout for the existing
SSD bridge (`blk.N.ffn_{gate,up,down}.EEE.weight`). For that layout, use
`--ssd-moe` and optionally `--hot-experts /path/to/hot_experts.json`. The hot-expert
file uses the repository's existing `{"gpu_experts": [[layer, expert], ...]}`
format. Packed GGUFs also work directly in SSD mode; a split copy is unnecessary.
See the current deployment guide for `--memory-budget-mib` and runtime reserves.
No machine paths, model weights, or private cache configurations are
included in this contribution. Conversion from Hugging Face is outside this port.

`--ar` runs the same target greedily without speculation. `--tokens-file path`
records emitted token IDs as a JSON array for comparison. `--audit-state-copies`
checks history-to-live copies byte for byte; this diagnostic causes additional
device-to-host transfers and should be omitted for timing measurements.

The executable does not apply a chat template. Supply an already rendered prompt
when using an instruction or thinking model. It defaults to stopping at EOG;
`--ignore-eos` permits fixed-length correctness comparisons.

## State handling

At the start of a round, the committed prefix has been decoded and the target
has sampled a root token which has not yet entered the cache.

1. Capture the prefix's recurrent component with
   `LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY`.
2. Run sparse draft steps using the shared memory.
3. Restore that recurrent snapshot and trim the draft-written attention KV tail.
   Verification therefore starts from the target prefix, not the draft state.
4. Verify `[root, draft_0, ..., draft_k]` in one target batch. Each recurrent layer
   writes convolution windows to `history_r` and DeltaNet matrices to `history_s`
   after each input token.
5. Compare the draft tokens against the corresponding target greedy predictions.
   If `a` draft tokens are accepted, commit history row `a`, representing `a + 1`
   verified input tokens including the root. Copy both recurrent state types,
   update recurrent position/source metadata, and remove later attention KV.
6. The correction or bonus token becomes the next root. It is not present in
   the previous round's committed state until decoded in the next round.

Zero draft acceptance still commits the known root. A one-token verification
writes the live state directly and must not copy an uninitialized history row.
All history operations return failure on unsupported capacity, lifecycle,
sequence, or position use. Capturing across microbatches is explicitly rejected.

The private prototype's older restore-and-replay commit path is not needed here.
Prefix snapshots before drafting and restoration before verification still cost
time and memory. Only the accepted-prefix **replay** is eliminated.

Despite its inherited `build_delta_net_chunking` name, the compatibility graph
unrolls DeltaNet recurrence token by token. Target verification is batched; this
port does not claim a parallel DeltaNet scan implementation.

## Memory cost

History scales with `capacity × sum_layers(bytes(conv_state) + bytes(delta_state))`.
In the earlier 80B-A3B Orin prototype, capacity 9 used 711,327,744 bytes
(678.4 MiB), and committing one prefix copied 79,036,416 bytes (75.4 MiB).
The per-round host snapshot adds roughly another recurrent-state copy. Exact
allocation depends on the model and state types and is logged at runtime.
History is included in memory accounting. These historical sizes are not a
throughput result for the public port.

## Reproducible correctness checks

Python with NumPy is needed only to create a deterministic, random-weight tiny
GGUF fixture. No model downloads are performed. CMake registers these tests when
it finds a Python interpreter that can import NumPy.

```bash
ctest --test-dir build-next -R qwen3next --output-on-failure

# Explicit CUDA execution of the same fixture:
build-next/bin/test-qwen3next-state build-next/tests/qwen3next-tiny-f32.gguf 99
python3 tests/test-qwen3next-equivalence.py \
  build-next/bin/llama-speculative-next build-next/tests/qwen3next-tiny-f32.gguf 99
```

Fixtures cover F32, F16, and BF16 matrix weights. Repeat the explicit CUDA checks
above with `qwen3next-tiny-f16.gguf` and `qwen3next-tiny-bf16.gguf` to cover those
types. The state test compares batched logits against serial AR at every retained
position, both recurrent state types after commit, and three subsequent correction
steps. It tests every prefix of verification lengths 1, 2, 4, 9, and 32, including
root-only (zero accepted draft tokens), partial, and full retention. It also
checks snapshot restoration after a sparse draft and rejects invalid capacities,
multiple sequences, discontinuous positions, history reuse, and split batches.
Floating-point comparisons use `atol=2e-4, rtol=2e-3`; this is not a bitwise
CPU/CUDA equivalence claim. These detailed state/logit comparisons use F32 KV to
isolate history correctness from F16 cache rounding. The eight-layer fixture
includes recurrent layers after a full-attention layer and a 64-token prefill
comparison. The executable-level test below uses the default F16 KV cache.

The executable-level test compares 40 greedy token IDs against AR for draft
depths 0, 1, 3, 8, and 31 and draft expert counts 1 and 2. The random fixture
tests implementation correctness, not language quality or benchmark performance.
Quantized weights, fused recurrent projections, partial/multi-GPU offload, and
broader sampling modes have not been qualified by these fixtures. Real 80B model
validation beyond the short smoke test below remains a separate release
qualification; the earlier prototype's logs alone do not validate this code.

## Port validation on Jetson Orin (2026-10-17)

- CPU-only build: all nine fixture/state/greedy-equivalence tests passed, covering
  F32, F16, and BF16 matrix weights.
- CUDA build: the same nine tests with an explicitly CPU-only device selection,
  plus the existing MoE reuse, MoE runtime-reuse, and DFlash regression tests,
  passed (12/12).
- Explicit GPU execution (`-ngl 99`): state and greedy-equivalence tests passed
  for all three fixture weight types (6/6).
- The local preconverted `Qwen3-Next-80B-A3B-Thinking-experts-F16.gguf` completed
  both SSD-backed self-speculation and AR for raw prompt `Hello`, with four output
  tokens. Both produced token IDs `[11, 358, 1184, 311]`. Self-speculation used
  draft depth 1, draft top-k 2, and audited history copies; both proposed tokens
  were accepted across two rounds. History capacity 2 allocated 158,072,832 bytes.

The 80B smoke test checks a real split-expert load and a short matching output;
it is not a benchmark or a claim of full-model equivalence across prompts.
Disabling eager mmap population for Next SSD mode was necessary to preserve
demand paging rather than prefaulting the entire large expert file.

## Opt-in real-model comparison matrix

```bash
python3 tests/validate-qwen3next-real.py \
  --binary build-next/bin/llama-speculative-next \
  --model /path/to/qwen3-next-split-experts.gguf \
  --ssd-moe --tokens 32 --output /path/to/new-validation-directory
```

The output directory must not already exist. This test performs twelve separate
process runs: three raw prompts (science, code, and arithmetic), each with one AR
reference and draft depth/top-k combinations `(1,1)`, `(4,2)`, and `(8,4)`.
Every run uses greedy decoding, the same context/microbatch/thread settings, and
fixed-length output. Model/binary size and modification time are checked between
cases to reject a changing test input. The script does not download weights.

`report.json` is updated after every case and distinguishes incomplete, failed,
and passed matrices. It records exact token IDs, the first mismatch if present,
per-round accepted lengths, and audited recurrent-state commit records. The
parser checks commit/proposal accounting as well as token equivalence. Raw native
logs and token files remain in separate case directories. These checks do not
constitute a performance benchmark.

## Same-environment type-check baseline

The separate [baseline comparison run](https://github.com/angerybob/S2-MoE/actions/runs/39019436391)
uses one Ubuntu/Python 3.11 dependency installation, Pyright 1.1.382, and the
original CI flags and configuration for both revisions. In that run, `orin`
(`48e952cfc`) and the comparison branch (`b1ba2f11e`) both report **17 errors and
1 warning**, with identical normalized diagnostics and no additions/removals.
The earlier repository CI reported 10 errors; these are separate runs and are
not presented as an identical historical dependency environment.

Raw JSON reports and the shared `pip-freeze.txt` are retained in the run artifact.
This establishes no new type-check diagnostics under the controlled comparison;
it does not make the repository-wide type-check pass. Unrelated old-script fixes
are excluded. The result is also recorded in the
[PR validation comment](https://github.com/angerybob/S2-MoE/pull/2#issuecomment-6232094471).

## Provenance

The architecture and recurrent-history implementation were extracted from the
maintainer's existing Orin prototype, with resident routing, utility scheduling,
benchmark orchestration, and other experiment-specific changes omitted. Existing
llama.cpp/S²-MoE MIT license notices remain applicable. This port introduces no
new inference dependencies and does not rebase the rest of the repository.

## Extended real 80B validation (2026-10-17)

The fixed-condition matrix at inference source revision `b1ba2f11e`
completed with **9/9** speculative sequences
matching their AR reference exactly. There were three AR runs and nine self-spec
runs, each producing 32 tokens. The model and executable stayed unchanged during
the matrix; subsequent commits only add validation evidence and automation/docs.

| Raw prompt | Draft depth | Draft top-k | Tokens | Exact AR match | Accepted/proposed | Zero/partial/full rounds |
|---|---:|---:|---:|---|---:|---|
| science | 1 | 1 | 32 | yes | 10/21 | 11/0/10 |
| science | 4 | 2 | 32 | yes | 22/37 | 2/4/4 |
| science | 8 | 4 | 32 | yes | 24/55 | 1/5/1 |
| code | 1 | 1 | 32 | yes | 13/19 | 6/0/13 |
| code | 4 | 2 | 32 | yes | 24/32 | 1/2/5 |
| code | 8 | 4 | 32 | yes | 24/44 | 2/3/2 |
| math | 1 | 1 | 32 | yes | 12/19 | 7/0/12 |
| math | 4 | 2 | 32 | yes | 24/28 | 1/0/6 |
| math | 8 | 4 | 32 | yes | 28/28 | 0/0/4 |

Across all speculative runs, the logs cover **31** zero-accept,
**14** partial-accept, and **57**
full-accept rounds. Each retained-prefix copy was audited, proposal/commit counts
conserved, and the accepted-prefix replay count remained zero. Raw prompts,
shared settings, exact token IDs, accepted lengths and native commit-audit records
are in the [machine-readable report](validation/qwen3next-80b-20261017.json).

This extends the earlier four-token smoke test for these three prompts and
configurations. It remains a correctness check for the tested 80B checkpoint and
conditions, not a performance benchmark or proof of all-input/all-backend
equivalence. These Next results do not qualify Qwen3.5-MoE; see its separate validation record.
