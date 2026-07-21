---
type: handoff
status: active
session: 19
date: 2026-07-20
description: "Shipped the disclosed temporal-prompt fix for the raw browser input `a bouncing ball`, repaired the parent JPEG media-type regression red-first, and passed the live CF++1 H100 JPEG-q90 release gate at 81 painted frames / 21 chunks / 21 ACKs with visible bounce replay and zero console warnings/errors."
branch: main
key_commits: []
prior_handoff: "session-18-motion-diagnosis-and-looping-replay"
---

# Session 19 Handoff — live temporal-prompt JPEG release

## TL;DR

- The Serving/Product temporal compiler now resolves the raw browser input
  `a bouncing ball` into a disclosed 106-word effective prompt specifying
  exactly three bounce arcs. Raw and effective prompts remain visibly distinct;
  exact/raw mode remains available. See [[CF1-Prompt-Conditioned-Motion]].
- The release blocker was not generation: `ProcessStreamingBackend` hard-coded
  `image/png` and rejected the real worker's valid boot-bound `image/jpeg`
  payloads. A parent validation hook plus the narrow CF1 JPEG override fixed the
  mismatch red-first. See [[Browser-Streaming-Transport]].
- The live CF++1 H100 JPEG-q90 path then completed end-to-end at 81 painted
  frames / 21 chunks / 21 presentation ACKs. Bounded replay visibly showed the
  ball leave the floor and return; browser warnings and errors were both empty.
- This is a user-visible Serving/Product release under
  [[ADR-003-serving-product-model-track-split]], not a model/inference
  experiment or a quality/performance authorization.

## What this session worked on

- **Disclosed temporal prompting** — the v2 resolver preserves subject count,
  action/count, camera, and framing while making temporal progression explicit;
  accepted output is 85–115 words and a one-time resolution identifier fences
  the disclosed effective prompt to the subsequent start.
- **JPEG parent/worker seam** — the default process backend retains strict PNG
  validation through `_validated_frame_media_type`; the CF1 backend overrides
  only its immutable `jpeg-q90-cpu-v1` boot profile and validates every JPEG
  payload before yielding a decoded chunk.
- **Live product release** — automatic prompt resolution, genuine H100
  generation, JPEG transport, browser paint/ACK, and local replay all completed
  as one user-visible path.

## Decisions made

- [[ADR-003-serving-product-model-track-split]] — Serving/Product owns this
  prompt/transport/playback path; agent1 retains CF++1 scheduling, KV-ring, and
  other model/inference experiments. AvatarForcing remains a decided NO-GO and
  was not re-evaluated here.

## State at session close

- [[State]] is the live truth.
- The dedicated pod, worker, and local tunnel remain intentionally running so
  the user can watch the accepted demo. No transient provider endpoint or
  credential is persisted in the brain.
- The accepted run is bounded development evidence. It does not prove clean
  physics, a 7/10 quality result, sustained service, FPS, TTFF, or a public
  performance headline.
- No git commit was created.

## Verification evidence

- Prompt-resolution focused suite: **14/14**.
- Executable browser client suite: **21/21**.
- Parent/process/worker media-type suite: **39/39** locally.
- Exact red-first CF1 JPEG regression passed in the remote Python 3.12 runtime:
  `test_real_backend_accepts_only_its_boot_bound_jpeg_payloads`.
- Real in-app browser: raw `a bouncing ball` disclosed a 106-word effective
  prompt, then completed 81 painted frames / 21 chunks / 21 presentation ACKs.
- Direct replay observation showed the ball leave the floor and return; browser
  console warnings and errors were both empty.
- Actual-source-and-tests CLI consensus returned **3/3 SHIP** from the required
  Claude, Kimi, and Grok lineages. The panel agreed that the JPEG media-type fix
  is correctly placed and red-first covered.

## Likely next moves

- Keep the live page available while the user watches the shipped result.
- When the user is finished, close the server/tunnel/worker gracefully and stop
  the pod without persisting its transient endpoint.
- Pursue the next user-visible Serving/Product motion or throughput increment;
  do not duplicate agent1's model/inference schedule work.
- Make mid-generation refresh/cancel recoverable before treating the demo as a
  durable multi-session service: disconnect currently cancels the registry job,
  leaves the warm-only process backend `stopped`, and has no automatic re-warm.
  This did not affect the completed 81/21/21 release run.

## See also

- [[Handoffs]]
- [[State]]
- [[ADR-003-serving-product-model-track-split]]
- [[CF1-Prompt-Conditioned-Motion]]
- [[Browser-Streaming-Transport]]
- Prior: [[session-18-motion-diagnosis-and-looping-replay]]
