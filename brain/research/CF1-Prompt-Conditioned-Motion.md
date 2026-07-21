---
type: research
status: active
date: 2026-07-20
description: "Five exact same-worker CF++1 captures established that the short bouncing-ball failure was prompt-conditioned action failure, not frozen transport. The Serving/Product temporal compiler now resolves the raw browser input `a bouncing ball` into a disclosed 106-word prompt specifying three bounce arcs; its live CF++1 H100 JPEG-q90 run completed 81/21/21 and replay visibly showed the ball leave the floor and return. This bounded product fix does not qualify the resulting physics or authorize a model-quality claim."
anchors: ["bench/results/cf1-motion-diagnostic-20260720", "bench/results/cf1-motion-generic-wrapper-20260720", "bench/model_assets/cf1-effective-config-v1.json", "bench/prompt_resolution.py", "bench/static/streaming_demo.html", "bench/static/streaming_demo.js"]
related: ["[[State]]", "[[Browser-Streaming-Transport]]", "[[CF1-Latent-Pull-and-Smoke]]", "[[CF1-Development-Video-Artifact]]", "[[Pegasus-1.5-Video-Judge-Calibration]]", "[[ADR-001-quality-qualified-headline]]", "[[ADR-003-serving-product-model-track-split]]", "[[session-18-motion-diagnosis-and-looping-replay]]", "[[session-19-live-temporal-prompt-jpeg-release]]"]
---

# CF++1 prompt-conditioned motion

## Bottom line

The user's report that the raw prompt “a bouncing ball” produced balls that did
not bounce is reproducible in the generated frames. It is not a frozen
WebSocket, duplicate frame, browser decode, or camera-only-motion failure. The
exact same warm worker produces large visible motion when the prompt specifies
the temporal action, impact, rebound, repetition, scale of travel, subject
count, and camera constraints. Changing the seed also changes the path and
amplitude.

Serving/Product now mitigates that terse-input failure with a disclosed,
versioned temporal-prompt compiler. In the live browser, the raw input
`a bouncing ball` resolved to a visible 106-word effective prompt specifying
exactly three bounce arcs. The subsequent CF++1 H100 JPEG-q90 request completed
81 painted frames / 21 chunks / 21 presentation ACKs, and bounded replay
visibly showed the ball leave the floor and return with no console warnings or
errors. This is a complementary product-layer workaround under
[[ADR-003-serving-product-model-track-split]], not a change to the CF++1 model
or inference schedule.

That result does **not** establish good motion quality. The successful detailed
clips bounce, but drift laterally, change scale/shape, clip out of frame, and
reverse nonphysically. This is a development diagnosis of prompt sensitivity
and action obedience, not a quality or performance authorization.

## Exact capture matrix

The diagnostic captured browser-WebSocket PNG payloads before browser decode
from five jobs on one already-warm worker. Every job completed the exact fixed
topology of 21 chunks and 81 frames. Within each clip all 81 PNG hashes were
unique, the saved bytes matched the manifest hashes, and no PNG hash overlapped
between clips. Deterministic 16-fps H.264 clips and contact sheets live under
`bench/results/cf1-motion-diagnostic-20260720/` and
`bench/results/cf1-motion-generic-wrapper-20260720/`.

“Adjacent MAD” below is mean absolute RGB pixel difference between neighboring
frames on the 8-bit scale. Ball spans are the observed range of its tracked
bounding-box center, not a learned quality score.

| Capture | Seed | Adjacent MAD | Ball center span | Direct observation |
|---|---:|---:|---:|---|
| `short-seed7` | 7 | 2.229 | x=14 px, y=4 px | No bounce. Center y remained 290–294 px and bottom edge 464–472 px; the ball stayed on the floor while its surface rotated/morphed. |
| `expanded-seed7` | 7 | 5.050 | x=110 px, y=152 px | Definite repeated vertical motion, but with lateral drift, scale change, deformation, and nonphysical reversals. |
| `expanded-seed20260719` | 20260719 | 5.541 | x=64 px, y≥220 px | Stronger bounce path, frequently clipped above the frame and still physically weak. |
| `upstream-rich-seed7` | 7 | 6.891 | n/a | Clear body, arm, and leaf motion; the requested spin/camera-circle action was only partially followed. |
| `generic-wrapper-seed7` | 7 | not scored | visually negligible vertical travel | Adding a generic “make the action obvious in one continuous five-second shot” instruction still left the ball planted. |

The background was mostly fixed in the ball clips. The changed bytes and
measured motion are object-level morphing/movement, not merely a camera move.
The explicit prompt at the same seed changes vertical travel from 4 px to 152
px, which is the cleanest controlled evidence that semantics—not delivery—are
driving this failure.

## Pegasus corroboration is advisory

Pegasus 1.5 independently analyzed the exact captured clips. It returned 0/100
motion for `short-seed7` and described the ball as static with no bounce,
agreeing with the 4 px measured vertical span. It also recognized clear motion
in the expanded-prompt clips, agreeing with the measured 152 px and ≥220 px
travel and the direct frame inspection.

This agreement is directional corroboration, not evaluator qualification. On
one expanded seed Pegasus over-credited the action and supported its account
with physically impossible timestamps when checked against the exact 81-frame
clip. That failure is consistent with the family's existing calibration NO-GO.
For this diagnosis, use provider analysis only as a second opinion and resolve
disagreement against manifest-bound frames, pixel metrics, tracked geometry,
and human inspection. Do not average it into a quality score.

## The detailed prompt that moved

The successful seed-7 diagnostic used this exact user-visible prompt:

> A locked-off medium shot shows one red rubber ball on a smooth white floor.
> The same ball repeatedly falls straight down under gravity, visibly
> compresses at each impact, rebounds upward, slows at the top of the arc, then
> falls again, completing three distinct bounce cycles. Its center rises and
> falls by about half the frame height while its contact shadow expands on
> impact and shrinks as it rises. Exactly one ball remains visible throughout;
> the floor and camera stay stationary in one continuous shot.

This diagnostic prompt established the useful temporal structure. The live
browser now keeps the raw input visible, discloses the resolved effective
prompt before generation, and offers exact/raw mode rather than presenting a
rewritten prompt as raw model behavior.

## What the source and prompt distribution say

- The exact/raw browser path sends the submitted prompt unchanged through the
  start command, registry, worker protocol, and generation call. With automatic
  resolution enabled, the browser first discloses an effective prompt; that
  exact resolved text then crosses the same unchanged generation path under a
  one-time resolution identifier. No hidden negative prompt is inserted.
- The frozen adapter's UMT5 tokenization/preprocessing matches the pinned
  upstream path. The official one-step demo also passes its selected prompt
  directly; the diagnosis is not explained by a missing public-demo wrapper.
- The 100 official demo prompts audited for this investigation contain 63–125
  words, with median 81 and mean 84.9. The failing terse prompt is far outside
  that demonstrated distribution.
- `bench/model_assets/cf1-effective-config-v1.json` points training data at
  `prompts/vidprom_filtered_extended.txt`, further evidence that the distilled
  model was trained around extended descriptions rather than three-word
  actions.
- The same training config contains `negative_prompt` and
  `guidance_scale`, but the pinned one-step inference path does not consume
  them. The negative string belongs to the distillation/training setup, not an
  omitted inference CFG feature. Injecting it at inference would be an
  unvalidated implementation drift, not a fidelity repair.

This evidence explains why a detailed prompt can help, but it does not excuse
poor terse-prompt behavior. A useful product should either improve that
behavior or disclose the model's prompt requirements honestly.

## Product temporal compiler

The product now implements prompt expansion as a distinct Serving/Product
component rather than hiding it inside the model path:

1. The page shows the user's raw input and the effective prompt separately
   before generation, and exact/raw mode remains available.
2. The resolver is versioned and its one-time resolution identifier binds the
   disclosed text to the subsequent start instead of permitting an unshown
   rewrite.
3. Strict parsing accepts only 85–115 whitespace-delimited words. The prompt
   contract preserves subject count, identity, requested action/count, camera,
   and framing while making temporal progression explicit; failed resolutions
   are not cached.
4. For the exact raw input `a bouncing ball`, the accepted live resolution was
   106 words and specified one ball, a fixed camera, opening and closing holds,
   and exactly three bounce arcs.
5. The live CF++1 H100 JPEG-q90 run completed its 81/21/21 presentation
   contract, and replay visibly showed vertical travel away from and back to
   the floor with zero console warnings/errors.

This crosses the user-visible motion release gate for this bounded example. It
does not establish clean physics, broad prompt robustness, a 7/10 quality
result, or a performance headline. The compiler must remain separately
observable and evaluable because better displacement is not automatically
better video.

## Remaining uncertainty

This matrix has two detailed seeds, one short seed, one rich non-ball prompt,
one failed generic wrapper, and one live compiled browser run. It establishes a
causal prompt-conditioning signal and a bounded product fix for this example,
not a broad prompt-response curve. Future Serving/Product work should test
registered action classes and multiple fixed seeds while agent1 retains
ownership of model/inference experiments. None of these captures authorize a
7/10 quality claim, an FPS/TTFF claim, or replacement of the existing evaluator
protocol.
