---
type: research
status: active
date: 2026-07-19
description: "The February 2025 Unsloth hiring puzzles form a coherent full-stack ML-systems curriculum—fused NF4 dequantization, FSDP2 QLoRA, graph-break-free compilation, OSS work, and chunked recomputation—but the floating environment and hardware-specific gates are not reproducible in July 2026. Task E's memory-efficient autograd pattern has the highest durable research value."
source_url: "https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH"
anchors: ["roofline/roofline.py#report_measured_h100_gate", "bench/harness.py#BenchRecorder", "PLAN.md"]
related: ["[[Research]]", "[[State]]", "[[ADR-001-quality-qualified-headline]]"]
---

# Unsloth puzzles — durable systems study

## Bottom line

The linked `Unsloth_Puzzles.ipynb` is not a tutorial, paper, or answer key. It is
Unsloth AI's now-closed February 2025 hiring challenge: 38 cells containing five
deliberately incomplete engineering problems across GPU kernels, distributed
training, PyTorch compilation, OSS contribution, and autograd memory design.

Its durable question is whether one engineer can reason across the complete
efficient-training stack—from packed weights and device memory traffic through
compiler capture and distributed execution to loss-level recomputation.

As of 2026-07-19, the concepts remain strong but the notebook is not a stable
benchmark. Its environment floats almost every load-bearing dependency, its
baseline moves with Unsloth `main`, its thresholds are tied to shared T4
runtimes, and its FSDP2/`torch.compile` assumptions predate substantial upstream
work. Treat it as a curriculum and design exercise, not as a current submission
path or reproducible performance claim.

Five independent model families reviewing the extracted requirements agreed
that **Task E, generalized memory-efficient backpropagation, has the greatest
durable learning and research value**. Task A is the strongest runner-up for
kernel engineering. This note reconciles that review with the notebook itself
and current primary documentation; reviewer errors are recorded explicitly
below rather than repeated as fact.

## Provenance and status

- Raw Drive download resolves to an unauthenticated `.ipynb` with 38 cells.
- Colab metadata traces the source template to Unsloth's
  `Llama3.1_(8B)-Alpaca.ipynb` notebook.
- The live notebook was edited after the recruiting round to say the challenges
  are closed. Rolling hiring language remains, but the puzzle submission route
  should be considered closed.
- The overview gives Task B 12 points while its detailed rubric caps it at 10:
  advertised total 57, detailed-rubric total 55.
- Public community solutions report plausible Task-A speedups and Task-E memory
  reductions, but the notebook contains no official answer key.

## Challenge map

| Task | Actual systems problem | Durable value |
|---|---|---|
| A — NF4 Triton | Fuse nested scale reconstruction and packed 4-bit weight decoding without an intermediate scale buffer | High for GPU-kernel craft; specific baseline is stale |
| B — QLoRA + FSDP2 | Define correct sharding semantics for frozen quantized state plus trainable adapters | High production relevance; API-coupled |
| C — `torch.compile` | Remove graph breaks and guard churn across quantization, PEFT, attention, checkpointing, and loss | High compiler-debugging value; exact failures drift by release |
| D — OSS issues | Reproduce, fix, validate, and upstream work in a live project | Good hiring signal; static issue/bounty list is obsolete |
| E — memory-efficient backprop | Chunk a large projection, discard logits, and recompute each chunk inside backward using autograd | Highest durable research value |

## A — fused NF4 double-dequantization

### Representation being decoded

BitsAndBytes NF4 uses two packed 4-bit codebook indices per byte, typically a
scale per 64 weights, and compressed statistics in which those scales are
themselves quantized as `uint8` in second-level blocks (the notebook asserts
block size 256), with another scale and an offset. Conceptually:

```text
block_scale = nested_code[q_scale] * nested_absmax + offset
weight       = nf4_code[nibble(q_weight)] * block_scale
```

The requested Triton kernel must do both reconstruction stages in one launch,
keeping the small lookup tables and scales close to the execution units and
never materializing the first-level scale array as a large temporary tensor.

### What the task tests

- Exact packed-nibble order and the 16-entry NF4 codebook.
- Distinct 64-weight and 256-scale indexing domains.
- Coalesced loads, masks, launch geometry, register pressure, and cache policy.
- Correct FP16/BF16 conversion and parity with BitsAndBytes/Unsloth.
- Whether a bandwidth/launch-bound primitive can beat an optimized baseline,
  not whether a new quantization algorithm can be invented.

### Benchmark audit

The notebook reports about 5.32 seconds for Unsloth and 5.59 for PEFT. A 1.15x
pass therefore needs roughly 4.63 seconds. It performs two warm-up passes, then
runs 1,000 iterations over each of three model shapes. Each iteration
dequantizes the up, gate, and down projections and synchronizes after every
dequantization—roughly 9,000 synchronized kernel calls in the aggregate result.

This deliberately includes launch and synchronization latency, not just kernel
bandwidth. What is missing is repeated-run variance, percentile reporting,
clock/thermal controls, and an immutable baseline version. The correctness
harness is stronger: it compares both full MLP output and each dequantized
matrix, including dtype and stride.

The required Tesla T4 is a **16 GB Turing, compute capability 7.5** device. It
does not provide native Ampere-style BF16 tensor-core execution. BF16 storage,
conversion, or emulated operations may work depending on the CUDA/PyTorch path,
but the notebook calculates `HAS_BFLOAT16 = major >= 8` and then ignores it
while including BF16 tests. That is an unresolved protocol contradiction.

## B — QLoRA under FSDP2

The starter's `device_map="auto"` is not FSDP2. It is module/device placement
and conflicts with the normal process-per-GPU, local-rank execution model FSDP
expects. A real design needs process-group initialization, rank-local device
selection, bottom-up `fully_shard`, an optimizer created after sharding, a
deterministic distributed sampler, and explicit DTensor save/load handling.

The hard issue is representation: BitsAndBytes parameters are tensor/parameter
subclasses carrying packed bytes plus quantization metadata. Arbitrarily
sharding the bytes can separate them from their block scales or split at invalid
quantization boundaries. A pragmatic design may replicate the roughly 4–5 GB
quantized base on each 16 GB T4 while distributing trainable LoRA state and
activations, but that is not equivalent to proving that the complete quantized
model is safely sharded. A submission must state which state is replicated,
sharded, reconstructed, and optimized.

"Same loss as single GPU" is also underspecified. Meaningful equivalence needs
the same effective global batch, sample order, accumulation, loss reduction,
dropout RNG, initialization, schedule, and precision policy, followed by
per-step loss/gradient or adapter-weight comparison under stated tolerances.

Historically this was a real frontier task: Hugging Face Accelerate's initial
FSDP2 PR merged on 2025-03-27, more than a month after the challenge appeared.
Current tooling lowers the integration burden, but quantized-state semantics
remain the valuable part of the exercise.

## C — graph-break-free compiled QLoRA

The starter uses `torch.compile(fullgraph=False, dynamic=True)` while demanding
no graph breaks. `fullgraph=False` permits eager fallback, so it cannot prove the
stated property. Current PyTorch guidance uses `fullgraph=True` as the diagnostic
guarantee: the selected region either captures as one graph or fails at the
first unsupported operation.

Likely break/guard sources include quantization metadata and custom operators,
tensor subclasses, PEFT wrappers, rotary/attention cache construction,
checkpoint recomputation, trainer-side Python, and loss extraction. Compiling
only `LlamaMLP.forward` leaves attention, normalization, quantized projections,
and loss outside the proof despite the rubric scoring them.

The durable technique is regional compilation: isolate repeated high-value
regions; make unsupported operations explicit custom ops with shape/fake
implementations where required; stabilize dynamic-shape guards; prove capture
with `fullgraph=True`; then measure unique backend compilations, eager/compiled
gradient equivalence, compile time, steady throughput, and peak memory. The
notebook's "about 30 compilations" threshold is meaningless without a pinned
PyTorch release and specified input-shape distribution.

## D — live OSS work

This tests reproduction discipline, repository navigation, maintainer
communication, compatibility testing, and small reviewable changes—not a stable
technical primitive. Several items were already marked `DONE` in the notebook,
and issue/bounty status must be rechecked against live GitHub before any work.

The Unsloth source citation uses a moving `main` line number; current
`kernels/utils.py` has grown to multiple hardware-specific `fast_dequantize`
definitions, so the old line no longer identifies the intended implementation.
The BitsAndBytes citation is safer because it points at immutable commit
`86b6c37a8ad448230cedb60753f63150b603a112`.

## E — generalized memory-efficient backpropagation

For hidden states `X ∈ R^(T×H)` and vocabulary weights `W ∈ R^(V×H)`, logits
`Z = XWᵀ ∈ R^(T×V)` dominate memory when `V` is large. With batch 4, sequence
4096, and vocabulary 128K, FP16/BF16 logits occupy about 4 GiB; an FP32 upcast
needs about 8 GiB.

Partition tokens into chunks `X_i`. Forward computes `Z_i = X_iWᵀ`, applies a
reducible transformation such as cross-entropy, retains only the small reduced
result, and discards `Z_i`. Backward recreates each chunk under
`torch.enable_grad()`, recomputes the projection and transformation, asks
`torch.autograd.grad` for local gradients, multiplies by the incoming upstream
gradient, writes the corresponding `dX_i`, and accumulates `ΣdW_i`.

Peak logit memory falls from `O(TV)` to `O(CV)` for chunk size `C`, at the cost of
recomputing the projection/loss. This is related to Apple's Cut Cross-Entropy,
but CCE is more specialized and can tile/fuse the vocabulary computation rather
than materializing even a chunk's complete logits.

### Starter-code trap

Passing `nn.Linear` as a non-Tensor custom-Function input passes the Python
object by reference; it is **not pickled or copied**, contrary to one reviewer's
claim. The real defect is that non-Tensor inputs do not create differentiable
autograd edges. The custom Function's forward also does not preserve its inner
ordinary graph. A robust interface should pass `X`, `weight`, optional `bias`,
and any tensor transformation inputs explicitly, then return their gradients
from `backward`.

### Conditions for exactness

- Mean-reduced chunk losses must be weighted by valid-token counts, especially
  with padding or `ignore_index`; averaging chunk means is not generally exact.
- `backward(ctx, dY)` must multiply recomputed local derivatives by `dY`.
- Recomputed stochastic functions must preserve RNG state.
- Autocast/device state must match forward.
- Uneven final chunks and dynamic chunk size need direct tests.
- A decreasing loss does not prove correctness: compare output, `dX`, `dW`,
  bias gradient, and a non-unit upstream gradient against the unchunked path.

The method is "general" only for transformations separable by chunk or reducible
through composable sufficient statistics. A function that couples arbitrary
tokens or batch elements cannot automatically be chunked exactly.

## How the tasks compose

```text
A: fused NF4 reconstruction ──► C: compiler-friendly quantized regions
             │                                  │
             └──────────────► B: distributed QLoRA

E: remove vocabulary-logit activation wall ───► combines with B/C

D: upstream any successful primitive into the real project
```

A supplies a traceable primitive for C and the format knowledge needed to reason
about B. B and C are orthogonal scaling axes but interact through DTensor/FSDP
hooks. E attacks a different materialization boundary and can enlarge the
sequence/batch envelope for either. D tests whether experimental work can become
maintainable software.

## Reproducibility verdict

The install cell pins only `xformers==0.0.29.post3`; PyTorch, Triton,
BitsAndBytes, Accelerate, PEFT, TRL, Transformers, Unsloth, and
`cut_cross_entropy` float. Consequently:

1. The 1.15x baseline changes with installation date.
2. Xformers/PyTorch/CUDA ABI compatibility is not guaranteed.
3. FSDP2 APIs and Trainer integration drift.
4. Dynamo graph breaks and compilation counts drift.
5. Quantized tensor layouts and dispatch behavior may drift.
6. Shared Kaggle T4 availability, clocks, and thermal state are uncontrolled.

A credible revival would lock the complete environment and hardware, preserve
an immutable reference implementation, gate performance behind correctness,
record warm-up and measurement distributions, and separate compile/launch,
device-kernel, and end-to-end timings.

## Relevance to realtime-video

The model domain differs, but four principles transfer directly:

1. **Optimize materialization boundaries.** A removes an intermediate scale
   buffer; E removes full logits. The realtime-video analogue is eliminating KV
   concatenation/copy materialization and using bounded rolling state.
2. **Compiler wins require state regularity.** C's graph-break problem parallels
   the project's measured inability to use CUDA graphs while KV caches are
   dynamically concatenated and mutated. A preallocated ring is an enabling
   representation change, not merely a micro-optimization.
3. **Correctness precedes speed.** The NF4 kernel must match decoded matrices;
   FSDP must match gradients; chunked backward must match `dW`. This is the same
   contract as [[ADR-001-quality-qualified-headline]]: faster output is not a
   valid result if semantics or quality changed.
4. **Benchmarks must freeze the stack.** Floating code and aggregate wall time
   make a performance threshold decay. `[[State]]` and the measured H100 gate
   are stronger because they preserve commit, hardware, protocol, cold/warm
   distinction, full-output correctness, and quality beside throughput.

## Priority if revisited

For durable systems learning: **E → A → C → B → D**. For current production
distributed-ML integration: **B → C → E → A → D**. A full-stack implementation
should prove each layer independently before attempting A+C+B+E composition.

## Primary sources

- [Unsloth puzzles notebook](https://colab.research.google.com/drive/1JqKqA1XWeLHvnYAc0wzrR4JBCnq43HyH)
- [Unsloth repository](https://github.com/unslothai/unsloth)
- [BitsAndBytes February 2025 reference commit](https://github.com/bitsandbytes-foundation/bitsandbytes/commit/86b6c37a8ad448230cedb60753f63150b603a112)
- [Accelerate FSDP2 PR #3394](https://github.com/huggingface/accelerate/pull/3394)
- [PyTorch `fully_shard` documentation](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/distributed.fsdp.fully_shard.md)
- [PyTorch `fullgraph=True` guidance](https://github.com/pytorch/pytorch/blob/v2.11.0/docs/source/user_guide/torch_compiler/compile/programming_model.fullgraph_true.md)
- [Triton documentation](https://triton-lang.org/main/)
- [Apple, “Cut Your Losses in Large-Vocabulary Language Models”](https://machinelearning.apple.com/research/cut-your-losses)

## Review corrections retained as safeguards

The independent panel was directionally consistent but not uniformly factual.
The synthesis rejected these claims:

- T4 is not 12 GB Maxwell; it is 16 GB Turing.
- A non-Tensor `autograd.Function` argument is not automatically pickled or
  copied; it simply lacks a differentiable input edge.
- The notebook does warm up Task A twice before timing, so first-call Triton JIT
  is not part of the measured loop.
- The three large MLP weight matrices cannot all remain resident in T4's small
  L2 cache, so the benchmark is not purely a cache-hot microbenchmark.
- `dynamic=True` may create guards/recompilation pressure, but does not by
  itself imply a graph break.

These corrections are why the final conclusions rely on direct notebook
inspection and primary docs rather than majority vote alone.
