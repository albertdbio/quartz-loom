---
type: gotcha
status: active
date: 2026-07-19
description: "Rolling TAEHV context is measured in latent frames, but decoder output is RGB frames. A fixed 12-frame startup trim produced 73 instead of 81 frames; the recovered path trims 3 frames for block zero, then `prior_context_latents * 4`, yielding exactly `[1, 4 × 20]`."
anchors: ["PLAN.md", "bench/harness.py#BenchRecorder", "roofline/roofline.py#report_measured_h100_gate", "bench/generation_preflight.py#rolling_taehv_trim_frames"]
related: ["[[ADR-001-quality-qualified-headline]]"]
---

# Gotcha — rolling TAEHV must trim the context actually supplied

The upstream rolling-decoder path always removed 12 RGB frames from each
decoded block. That is correct only once the rolling context contains three
latent frames. During startup, fewer context latents exist, so the fixed trim
discarded generated output and produced a 73-frame video from an 81-frame run.

The exact recovered invariant is:

```text
rgb_frames_to_trim = 3                                  if block_index == 0
                     prior_context_latents * 4          otherwise
```

The first one-latent TAEHV decode returns four RGB frames, but only its final
frame belongs to the causal rollout, so startup must trim three. After startup,
the retained context grows from one to three latent frames; trimming four RGB
frames per retained latent naturally grows to 12 once that context is full.
The executable helper `rolling_taehv_trim_frames` pins the required context
state at every block and reproduces the release schedule `[1, 4 × 20]`. After
the original fix, every audited output contained all 81 expected frames.

Do not validate this path from fps alone: a frame-dropping decoder can look
faster while silently shortening the result. Assert exact output frame count,
inspect first/last-frame continuity, and score the full video.
