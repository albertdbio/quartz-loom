---
type: handoff
status: active
session: 12
date: 2026-07-20
description: "Added non-authorizing H100 capture, the exact 45-forward CF++1 latent pull loop, and bounded/full serial PNG smokes; local implementation is GO, while real H100 admission and generation remain NO-GO."
branch: main
key_commits: []
prior_handoff: "session-11-runtime-preflight"
---

# Session 12 Handoff — latent pull and bounded CUDA smoke

## TL;DR

- Added [[CF1-Latent-Pull-and-Smoke]]: the exact 21-block/45-forward CF++1
  generator, joined to rolling TAEHV and pinned torchvision PNG encoding.
- Added `scripts/cf-runtime-capture`, a separate stdlib-only observation path
  that always remains `authorizes_boot:false` / `ready:false`.
- Added `scripts/cf-cuda-smoke`: one block must produce one PNG and exit; 21
  blocks must produce 81 PNGs and cleanly finish both sessions.
- The candidate runtime still refuses. No real model load, CUDA forward, frame,
  video, Gemini/TwelveLabs upload, or GPU spend occurred.

## What changed

`bench/cf_cuda_generator.py` reproduces upstream seed/RNG, prompt, cache, noise,
denoising, and clean-context-refresh semantics. Block zero performs 4+1 model
forwards, each remaining block 1+1, for 45 total. Released latents are cloned
before cache refresh; producer events are recorded after it. Exact tensor,
pipeline, ownership, cache, and final-index checks fail closed.

`bench/cf_cuda_smoke.py` supports only one or 21 blocks. It writes exact PNG
hashes plus runtime/bootstrap/guard provenance, sanitized partial failure
manifests, and honestly labeled serial timing. The clock synchronizes away
runtime bootstrap, then begins before session initialization. It is not browser
presentation or the archived overlap benchmark.

`bench/cf_runtime_preflight.py` now separates collection from authorization.
Capture validates the exact checked-in lock bytes and records safe stdlib facts
without importing Torch or models, mutating the lock, or exposing unrelated
environment variables. Wheel/ABI/backend facts remain unresolved by design.

## Independent review

Consensus parent `ses_08069a872ffe7KezWVDntsRI5K` requested
`claude-opus,kimi-k3,grok` and timed out. Child recovery retained concrete
`grok-4.5` at `ses_08065cd0effcm37VK02UlWsJ5B`; Fable-5 and Kimi-K3 produced
zero output. There was no detected Fable rate limit, so no automatic Opus retry
was appropriate.

Local fallback session `8647e7f3-fa95-428b-8737-66ef4a677298` exposed concrete
`claude-opus-4-8`. Its two passes and Grok's review cost **$3.5000955** total.
All reproducible findings were fixed red-first. A final independent source
audit then reproduced two additional issues: `KeyboardInterrupt` could resume
dirty generator/decoder state, and async setup made the smoke clock ambiguous.
Generator init/pull/finish and decoder decode now poison on `BaseException`
while re-raising interrupts unchanged; runtime CUDA is synchronized before a
precisely scoped clock begins. The same auditor's post-fix verdict is local GO
with no remaining P0–P2 issue. The incomplete panel cannot clear project
NO-GO.

## Verification

- Focused generator/smoke/session/adapter/runtime suite: **72/72**.
- Generator/session/smoke subset after final audit fixes: **29/29**.
- Python `compileall`, repository JSON parsing, and Node browser tests: green;
  Node **4/4**.
- Bare system-Python discovery passes 283 tests but cannot import the 20-test
  WebSocket module because `aiohttp` is absent. An isolated temporary target
  installed the exact `aiohttp==3.13.5` requirement and the complete suite then
  passed **303/303** without changing the project environment.
- Local `scripts/cf-runtime-capture` exits 2 on absent `/proc/meminfo` while
  remaining explicitly non-authorizing.
- Local one-block `scripts/cf-cuda-smoke` exits 2 at runtime preflight and leaves
  no output directory.

## Operational blocker

The in-app RunPod console is open but signed out, and no provider CLI/API
credential is available locally. The old pod `ooprxl8l5c7c59` remains stopped;
its volume was previously recorded as persistent. This prevented an H100
capture, not a local implementation step. The user has enabled video uploads,
so the first complete generated artifact can go directly to TwelveLabs and
Gemini after the GPU proof; there is no reason to spend another call on the old
calibration media.

## Exact next sequence

1. Restore authenticated RunPod control, start the dedicated one-H100 worker,
   and obtain the immutable image digest from provider-observed metadata.
2. Run `scripts/cf-runtime-capture`; reconcile Python, complete package set,
   wheel/ABI, driver, and active backend into the lock. Independently review the
   inventory before changing `candidate` to `frozen`.
3. Run `scripts/cf-runtime-preflight` unchanged. Stop on any mismatch.
4. Run `scripts/cf-cuda-smoke --blocks 1` and decode-check the one PNG. The
   process must exit afterward even on success.
5. If and only if step 4 passes, run `--blocks 21`, assemble the 81 exact PNGs
   at explicit 16 fps, and upload that exact video to Gemini and TwelveLabs for
   development understanding checks.
6. Only then attach the real path to the persistent worker/handshake/browser
   boundary and measure presentation. Evaluator qualification, long-horizon
   policy, registered selection evidence, completed phase-gate review, and the
   sustained ≥60-second run remain separate blockers.

## See also

- [[CF1-Latent-Pull-and-Smoke]]
- [[CF1-H100-Runtime-Preflight]]
- [[CF1-Rolling-TAEHV-Session]]
- [[State]]
- Prior: [[session-11-runtime-preflight]]
