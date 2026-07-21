---
type: decision
status: accepted
date: 2026-07-20
description: "Serving/Product owns user-visible temporal prompting, transport, playback, and live throughput delivery; the isolated Model/Inference track owns CF++1 scheduling and KV-ring research, with AvatarForcing already rejected."
anchors: ["bench/prompt_resolution.py", "bench/streaming_process.py#ProcessStreamingBackend", "bench/cf_streaming_worker.py#CF1ProcessStreamingBackend", "bench/streaming_web_demo.py"]
related: ["[[CF1-Prompt-Conditioned-Motion]]", "[[Browser-Streaming-Transport]]", "[[State]]"]
---

# ADR-003 — Separate serving/product delivery from model/inference research

## Decision

From 2026-07-20 onward, real-time-video work is split into two isolated,
complementary tracks:

1. **Serving/Product** owns the temporal-prompt compiler, frame transport,
   browser playback, real end-to-end fps, and live user-visible delivery.
2. **Model/Inference** (agent1) owns inference schedules, the CF++1
   recent-clean-block KV-ring flag, and model/inference research.

The Serving/Product track must not re-evaluate AvatarForcing and must not start
parallel CF++1 inference-schedule experiments. Its temporal-prompt compiler is
a complementary serving workaround, not a competing model-track experiment.

## AvatarForcing disposition

The isolated Model/Inference track has already established an AvatarForcing
**NO-GO**: it measured 13.2 fps with motion rated 5/10, losing to a self-forcing
ablation at 14.7 fps with motion rated 7/10. AvatarForcing is also
avatar-specific and requires 1.92 seconds of look-ahead. Serving/Product treats
that evaluation as decided and does not spend another session deriving it.

## Session delivery gate

Every session must produce a user-visible motion or throughput improvement: a
live demo or a real fps/quality delta. A session that only hardens integrity
evidence, manifests, or hashes is a failed session.

Before starting any integrity, evidence, or manifest task, ask: **Does this
unblock a user-visible improvement right now, or is it only hardening
evidence?** If it is only evidence hardening, stop and ship the serving/product
improvement instead.

## Consequences

- Serving/Product prioritizes getting the proven temporal-prompt motion fix
  through the production worker and into the live browser demo.
- Model/inference findings are consumed as inputs rather than duplicated.
- Evidence work is justified only when it directly blocks an immediate live
  motion or throughput result.
