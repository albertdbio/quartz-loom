---
type: research
status: active
date: 2026-07-20
description: "The exact 45-forward CF++1 latent loop has passed one-block and complete 21-block/81-PNG proofs behind the frozen H100 authorizer; deterministic video, dual-provider development upload, persistent-worker reuse/poison, and real-browser presentation also completed while all timings remain non-benchmark."
anchors: ["bench/cf_cuda_generator.py#CF1LatentPullSession", "bench/cf_cuda_generator.py#pull", "bench/cf_cuda_smoke.py#run_cf1_cuda_smoke", "bench/cf_cuda_smoke.py#build_png_encoder", "bench/cf_runtime_preflight.py#runtime_capture_report", "bench/cf_runtime_preflight.py#capture_runtime_environment"]
related: ["[[Research]]", "[[State]]", "[[Pinned-CF1-CUDA-Bootstrap]]", "[[CF1-H100-Runtime-Preflight]]", "[[CF1-Rolling-TAEHV-Session]]", "[[session-12-latent-pull-and-smoke]]"]
---

# CF++1 latent pull and bounded CUDA smoke

## Bottom line

`CF1LatentPullSession` ports the pinned CF++1 generation loop one latent block
at a time, while `run_cf1_cuda_smoke` joins it to rolling TAEHV and the exact
runtime's torchvision PNG encoder. The frozen authorizer passed and this path
has now executed on the dedicated H100.

The default one-block smoke produced one decodable 832×480 PNG and exited. The
complete mode then produced 21 chunks and 81 exact PNGs under the canonical
`[1, 4 × 20]` schedule; manifest SHA is
`d363fbe1b65b167a5cf731aa194b00676f23a4d6cdcdaa0f89cfe8a7c960ba99`.
Those bytes were deterministically assembled into the development MP4 and
uploaded once to Gemini plus TwelveLabs. This is a real boot/generation proof,
not protocol freeze, quality, performance, browser latency, or sustained-run
evidence.

## Exact generator contract

The generator requires the verified `cf1-rolling-taehv-v1` runtime, one
exclusive pipeline lease, `independent_first_frame=False`, one latent per
block, 30 transformer blocks, a 1,560-token frame sequence, global attention,
and zero context noise. It reproduces the recovered short-run schedule:

- block zero: four denoising forwards plus one clean context-cache refresh;
- blocks 1–20: one denoising forward plus one context refresh each;
- total: 45 model forwards for 21 latent blocks.

Seed handling follows upstream: Python, NumPy, Torch, and all CUDA generators
are seeded, while a separate CUDA-local generator creates the full
`1×21×16×60×104` bfloat16 initial-noise tensor. The first block performs exactly
three global `randn_like` schedule transitions. Text conditioning must be
`1×512×4096` bfloat16 on the verified device.

Cache reuse resets cross-attention `is_init` and zeroes the KV end indices
without clearing K/V storage. A released denoised latent is detached and cloned
before the context refresh can reuse or overwrite upstream storage. Its producer
event is recorded only after that refresh completes on the generation stream.
Finalization requires every KV end index to equal 32,760.

## Ownership, interrupts, and poison

Generation and decode are non-reentrant. A clean 21-block finish releases the
generator's pipeline lease. Any failure after mutation begins, including
`KeyboardInterrupt`, `SystemExit`, or another `BaseException`, marks the
generator pipeline or decoder poisoned and preserves ownership so the future
worker must kill and reap the process. Interrupts are re-raised unchanged;
ordinary implementation errors retain their typed wrapper. This distinction is
important because a failed forward can advance KV indices before raising.

The monotonic interrupt rule was added after an independent probe advanced a
cache to 1,560, raised `KeyboardInterrupt`, and demonstrated that the original
`except Exception` boundary could resume dirty block zero. Red-first regressions
now cover initialization, post-cache-mutation pull, generator finalization, and
TAEHV decode.

## Bounded executable proof

`scripts/cf-cuda-smoke` accepts only `--blocks 1` or `--blocks 21`:

- one block writes one PNG, reports `bounded-first-chunk`, marks the runtime
  non-reusable, and requires the CLI process to exit;
- 21 blocks write 81 PNGs, finish decoder then generator, and report clean
  reuse eligibility.

Every frame is encoded by pinned torchvision at PNG compression level 1,
checked for CPU uint8 `3×480×832` ownership and PNG boundaries, written under a
new output directory, and SHA-256-bound in `manifest.json`. Prompt text is never
persisted; only its SHA-256 appears. Once an output directory exists, ordinary
failures and interrupts attempt to leave a sanitized failed manifest with the
partial frame hashes and bootstrap/runtime/guard identities.

This is a serial measurement, not the archived overlap path. The clock begins
after an explicit runtime-device synchronization and before session
initialization. It therefore includes noise creation, prompt encoding, cache
initialization, decoder construction, generation, context refresh, rolling
decode, D2H, PNG encode, and frame writes; it explicitly excludes runtime/model
bootstrap, encoder construction, directory creation, and manifest writing. The
manifest calls the first boundary `first_chunk_encoded_s` and the filesystem
boundary `first_frame_written_s`; neither is browser presentation.

## Runtime capture, freeze, and replay

`scripts/cf-runtime-capture` remains a separate non-authorizing observation
command: it cannot edit or freeze the lock or turn observed values into desired
ones. The dedicated H100 capture was independently reconciled into evidence SHA
`8209043b…` and frozen-lock SHA `d4d163d6…`; the unchanged authorizer then passed.

The first real execution exposed an important environment-integrity bug: normal
Python imports created 1,649 bytecode files inside the frozen runtime tree.
Their exact count and hashes were identified before removal. After cleanup, the
authorizer was replayed and the full smoke completed with zero changed runtime-
environment files. Launches now disable bytecode before project imports, and
the tree equality check remains part of acceptance rather than being waived.

## Review and completed run

The requested `claude-opus,kimi-k3,grok` consensus parent
`ses_08069a872ffe7KezWVDntsRI5K` timed out. SQLite recovery retained a concrete
`grok-4.5` analysis from child `ses_08065cd0effcm37VK02UlWsJ5B`; concrete
`claude-fable-5` and `moonshotai/kimi-k3` children produced zero output. Fable
did not report a rate limit, so automatic Opus fallback was not triggered.

A complete-source local fallback used concrete `claude-opus-4-8`, session
`8647e7f3-fa95-428b-8737-66ef4a677298`. Its poison, timing-scope, partial
manifest, naming, and provenance findings were reproduced red-first and fixed.
A final independent audit found the BaseException poison and asynchronous
timing-origin defects above; those too were reproduced red-first before their
fixes. The implementation is locally reviewed, but an incomplete panel cannot
clear the project gate.

That exact sequence completed: capture/reconcile/freeze, unchanged admission,
one-block proof, 21-block/81-PNG proof, deterministic 16-fps assembly, full
decode, and one exact-byte development upload to both providers. The worker
acceptance later repeated the same generator/decoder boundary twice on one warm
PID/instance and proved forced-death poison/reap.

The real CF++1 browser path now passes separately: 81 frames, 21 chunks, and 21
post-paint ACKs reached a visible canvas with server completion and zero console
errors. None of the smoke's filesystem or serial PNG-readiness times becomes a
performance result merely because presentation succeeded.
