---
type: handoff
status: active
session: 10
date: 2026-07-20
description: "Added and adversarially pinned the local CF++1 rolling-TAEHV CUDA session core, including full runtime identity, explicit stream/event ownership, exact dynamic trims, synchronized D2H, immutable raster validation, completion, and poison semantics."
branch: main
key_commits: []
prior_handoff: "session-9-pinned-assets-and-cuda-bootstrap"
---

# Session 10 Handoff — Rolling CUDA session core

## TL;DR

- `bench/cf_cuda_session.py` now implements the exact 21-block rolling-TAEHV
  decode contract around a verified pinned runtime: bfloat16 input to float16
  decode, chronological three-latent context, trims `3/4/8/12…`, and releases
  `[1, 4 × 20]`.
- It binds exact source/lock/config/assets plus recomputed bootstrap identity and
  a five-module guard digest, waits on a required producer event, owns an
  explicit decode stream, uses `record_stream`, synchronizes after GPU
  postprocessing, then verifies CPU uint8 rasters and immutable bounded encoded
  payloads.
- This remains local mocked-core scaffolding. No real Torch/model load, CUDA
  session, generated frame, raster-encoder smoke, worker integration, browser
  generation, GPU use, or Gemini/TwelveLabs call occurred. The project stays
  NO-GO.

## What changed

- Added `RollingTaehvChunkDecoder` with fail-closed runtime provenance and
  explicit CUDA-device validation.
- Bound bootstrap identity to exact source, lock, effective config, observed
  assets, and the canonical guard-manifest digest of the adapter, session, generation
  preflight, model-asset preflight, and streaming-service modules.
- Implemented exact latent shape/device/dtype checks, explicit producer-event
  wait, decode-stream ownership, `record_stream`, rolling last-three context,
  canonical dynamic trims, and exact output shape/device/dtype checks.
- Moved D2H and encoding after a blocking postprocess event; encoded payloads
  must have raster magic, remain within the byte cap, and be immutable `bytes`.
- Added exact 21-chunk completion, retained-tail release, commit-after-success,
  and permanent poison on any post-start failure.
- Hardened the adapter so TAEHV is constructed with `checkpoint_path=None` and
  its pinned state is loaded with `weights_only=True`, `mmap=True`,
  `strict=True`, and `assign=True`.
- All new behavior was developed red-first; the independent decoder review's
  reproducible mocked-core mutations failed before their fixes.

## Review

Consensus parent `ses_080ea9c12ffejex9NyF4Yi7sk0` timed out without a
synthesis. Direct session recovery found concrete `claude-fable-5` child
`ses_080e6c4cbffewgbBPMTGzWaxMV` at GO-WITH-FIXES / 0.78 and concrete
`grok-4.5` child `ses_080e6c4c9ffeiwnSU0HNzAeM3r` at GO-WITH-FIXES / 0.82.
Kimi child `ses_080e6c4caffeh0tWtS1ApYcJPr` stalled with zero tokens. Fable
succeeded, so no Opus fallback was appropriate.

A separate local Claude CLI review requested Opus, returned GO-WITH-FIXES, and
mutation-tested the decoder. Every actionable mocked-core gap reproduced red
and was fixed/pinned. Its concrete model version was not exposed. These are
partial/fallback reviews, not completed three-voter consensus or phase-gate GO.

## Final observed verification

- Complete Python suite: **256/256**.
- Persistent process: **35/35**.
- Process + service + NDJSON + WebSocket: **113/113**; the same focused set
  without process is **78/78**.
- Asset preflight: **13/13**; CUDA adapter: **16/16**; CUDA session: **9/9**.
- Executable Node browser client: **4/4**.
- Python `compileall` and all repository JSON parses: green.
- Real asset gate: `ready: true`, exact source commit
  `8db419e341e5fc52542c0b2c4542728420ddfb4a`, lock SHA-256
  `0aee8671f8e3b30286b689a16f6f4a355f917772c16e599cec75a49e89057967`,
  all 11 assets verified, zero unexpected paths.
- No GPU or Gemini/TwelveLabs call was made, no secret was printed or
  persisted, and no commit was created. Consensus and external model review did
  run. The earlier fourth Pegasus upload succeeded after the permission change,
  but calibration remains failed.

## Honest remaining gates

- Runtime dependency versions and tokenizer sentinel IDs still float.
- Peak CPU RAM and VRAM are not proved; mmap/assign reduces copies but does not
  eliminate fully initialized FP32/CPU model residency. Establish a capacity
  plan or meta/empty initialization path before real loading.
- No real Torch/model load, CF++ generation loop, producer/decode CUDA event,
  encoded raster decodability/dimensions, or GPU frame exists locally.
- The real raster encoder and session are not bound into the worker handshake,
  provenance, pull loop, or browser path; poisoned-worker kill/reap remains to
  be integrated.
- Source verify-to-path-reopen TOCTOU is outside the trusted/no-concurrent-writer
  threat model but returns if bytes become remote, untrusted, or multi-tenant.
- Browser generation, finalist selection, evaluator/human qualification,
  confirmatory evidence, sustained timing, and completed review gates remain
  open. Pegasus is still `calibration-failed`; Gemini is calibration-pending.

## Next bounded experiment

After the memory plan and dependency/tokenizer pins, start one dedicated real
CUDA worker and run only a boot/session/frame smoke: load the exact tensors,
exercise real producer/decode event ownership, generate one latent through the
rolling decoder, encode and decode-check 832×480 output, and bind the
encoder/session into handshake provenance with poisoned-worker retirement. Do
not expand this into a benchmark, evidence sweep, or protocol freeze.

## See also

- [[CF1-Rolling-TAEHV-Session]]
- [[Pinned-CF1-CUDA-Bootstrap]]
- [[Streaming-Service-Boundary]]
- [[State]]
- Prior: [[session-9-pinned-assets-and-cuda-bootstrap]]
