---
type: gotcha
status: active
date: 2026-07-19
description: "A quality score qualifies a throughput result only when both use the same decoder execution path; batch and rolling TAEHV outputs are not interchangeable."
anchors: ["bench/quality_sweep.py#validate_run_manifest", "bench/reference/h100_stream_overlap_gate_recovered.py"]
related: ["[[Gotcha-Seed-Is-Not-Artifact-Provenance]]", "[[ADR-001-quality-qualified-headline]]", "[[State]]"]
---

# Gotcha — Quality must use the performance decoder path

TAEHV checkpoint identity is not enough to establish stack identity. A single
full-batch decode and a rolling three-latent decode exercise different temporal
context and trimming behavior; that difference can materially change the
video even when prompt, seed, checkpoint, and frame count match.

The exact decoder mode is therefore a manifest invariant. CF++ baseline and
repair candidates must declare `rolling-three-latent`; the SF4 reference must
declare its separately pinned stock-Wan path. The run manifest, latent hash,
media hash, codec, pixel format, and decoded-frame audit must all agree before
a quality result may accompany a throughput result.
