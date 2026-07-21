---
type: handoff
status: active
session: 18
date: 2026-07-20
description: "Separated the user's no-bounce report into a genuine terse-prompt action failure and a one-pass terminal-frame playback defect; captured five exact same-worker clips, obtained bounded Pegasus corroboration, added bounded local looping replay plus exact topology/security fences, restored the frozen runtime after a fail-closed 62-file bytecode-drift refusal, and passed exact 81/21/21 real-browser replay reacceptance with stable counters and zero console errors."
branch: main
key_commits: []
prior_handoff: "session-17-frozen-h100-generation-and-browser-acceptance"
superseded_by: "session-19-live-temporal-prompt-jpeg-release"
---

# Session 18 Handoff — motion diagnosis and looping replay

## TL;DR

- The user's manual “a bouncing ball” result really does not bounce. Exact
  WebSocket capture proved 81 unique changing frames, so transport is not
  frozen; the ball remains planted while its texture/surface morphs. See
  [[CF1-Prompt-Conditioned-Motion]].
- A detailed temporal prompt at the same seed produces a visibly bouncing ball,
  and another seed changes its path. Motion is strongly prompt-conditioned,
  but both “successful” clips have weak physics, drift/deformation, and
  clipping. A generic instruction suffix did not repair the terse prompt.
- The page had a separate playback defect: it painted each frame once, closed
  it, discarded history, and left frame 81 frozen after completion. The client
  now loops a bounded retained clip locally without changing original counters
  or sending replay ACKs. See [[Browser-Streaming-Transport]].
- Replay/topology/security changes pass **19/19** executable Node tests; the
  targeted Python WebSocket/server suites pass **27/27**. Grok's one completed
  external advisory found real gaps that were fixed red-first; Fable and Kimi
  timed out, so this was not a three-lineage consensus.
- Pegasus 1.5 independently returned 0/100 motion for `short-seed7`, calling it
  static/no-bounce, and recognized clear motion in the expanded prompts. It
  also over-credited one expanded seed with physically impossible timestamps,
  so it is corroboration only and cannot override frame inspection or metrics.
- The synchronized relaunch first failed closed because the static runtime
  evidence found exactly 62 generated `.pyc` files / 1,006,933 bytes across
  aiohttp and dependencies. Package versions and source bytes were unchanged.
  Only those caches were removed after exact path/count/byte validation; full
  evidence then returned to exact lock equality and the no-bytecode wrapper
  relaunched cleanly.
- Exact final static assets are synchronized and a clean worker/server passed
  real-browser replay reacceptance: complete 81/21/21 original counters,
  `replayState=replaying`, changing screenshots across 500 ms with counters
  unchanged, and zero console errors. No quality or performance claim is
  authorized.

## Exact diagnostic evidence

Five jobs ran on the same warm CF++1 worker. Each completed 21 chunks and 81
unique, manifest-matching PNGs; there was no cross-clip frame-hash overlap.
Deterministic 16-fps MP4s, contact sheets, frames, and capture manifests are in
`bench/results/cf1-motion-diagnostic-20260720/` and
`bench/results/cf1-motion-generic-wrapper-20260720/`.

| Clip | Adjacent 8-bit MAD | Ball-center span | Finding |
|---|---:|---:|---|
| `short-seed7` | 2.229 | x=14 px, y=4 px | No bounce; center y=290–294 and bottom edge y=464–472 while the ball surface morphs. |
| `expanded-seed7` | 5.050 | x=110 px, y=152 px | Definite repeated bounce, but lateral drift, scale change, deformation, and nonphysical reversal. |
| `expanded-seed20260719` | 5.541 | x=64 px, y≥220 px | Strong motion, frequently clipped above frame and still physically weak. |
| `upstream-rich-seed7` | 6.891 | n/a | Clear body/arm/leaf motion, but incomplete requested spin/camera circle. |
| `generic-wrapper-seed7` | not scored | visually negligible vertical travel | Generic “make the action obvious” boilerplate still leaves the ball planted. |

The background is mostly fixed in the ball captures. Prompt specificity changes
object travel from 4 px to 152 px at the same seed; this rules out a simple
transport freeze while retaining the terse-prompt failure as an honest model
limitation.

## Independent video-understanding check

Pegasus 1.5 analyzed the exact captured clips independently of the pixel and
ball-track measurements. It gave `short-seed7` a 0/100 motion result and
described the ball as static with no bounce, matching the 4 px vertical span.
It recognized obvious movement in both expanded-prompt clips, matching the
direction of the measured 152 px and ≥220 px spans.

That agreement is useful but bounded. For one expanded seed Pegasus
over-credited the physical action and supported its account with physically
impossible timestamps when checked against the exact frames. The family is
already calibration-failed and remains so. Use this result only as independent
directional corroboration; exact frames, hashes, pixel differences, tracked
motion, and human inspection remain authoritative for this diagnosis.

## Prompt-path finding

- Browser → registry → worker → generation sends the user's raw prompt
  unchanged. UMT5 preprocessing matches the pinned upstream implementation,
  and the official one-step demo also sends the selected prompt directly.
- All 100 official demo prompts audited here are 63–125 words, median 81 and
  mean 84.9. The effective training config points at
  `prompts/vidprom_filtered_extended.txt`. The failing three-word prompt is far
  outside the demonstrated prompt distribution.
- The training config's negative prompt and guidance scale are not consumed by
  the pinned one-step inference path. Adding inference CFG/negative prompting
  would be unvalidated drift, not restoration of an omitted upstream behavior.
- A future LLM prompt expander could be useful only as a disclosed, versioned,
  separately evaluated component that preserves and hashes both user and model
  prompts. The generic-wrapper failure shows why silent boilerplate is not a
  credible fix.

The detailed bouncing-ball prompt is now the page's worked default, alongside
an explicit warning that very short prompts may produce little motion. The page
does not silently rewrite submitted input.

## Client correction and hardening

The original presentation proof remains valid for one 81-frame paint/ACK pass,
but its terminal canvas was not a video player. The new client:

- retains encoded frames only after successful original presentation and loops
  them locally at 16 fps;
- caps each incoming chunk at 16 MiB and total retained replay bytes at 64 MiB;
- keeps stream and replay pacing, jobs, and epochs separate;
- leaves original rendered-frame, rendered-chunk, and ACK counters unchanged
  during replay and never sends a replay `presented` command;
- fences stale in-flight paints on replacement, pre-completion
  disconnect/error/cancel, and new-job epochs;
- preserves replay across a post-completion socket close;
- treats replay-only decode failure as replay degradation without erasing
  already-proven original completion;
- accepts only the fixed 21-latent/81-RGB job and exact `1 + 20×4` chunk index,
  count, and first-frame-offset topology before paint;
- fails closed on malformed chunks or premature completion; and
- exposes a validated user-controlled uint32 seed instead of only a hardcoded
  seed.

## Review and tests

The actual-file consensus request used standing aliases `claude-opus`,
`kimi-k3`, and `grok` in that order. Parent
`ses_07e3913c1ffeJA1r0XbUltbgDs` timed out after roughly 570 seconds:

- concrete Fable child `ses_07e38d318ffeKb3Tm59uxZn0TR` timed out with no
  output and no rate-limit signal, so automatic Opus fallback did not run;
- Kimi child `ses_07e38d315ffeMGw3O0GnCGfvyP` timed out without a final; and
- Grok 4.5 child `ses_07e38d314ffenrE2Ihkvz4kqyJ` returned BLOCK for
  pre-completion stale-paint races, absent expected-topology validation, an
  empty/tight replay possibility, missing negative tests, and no browser byte
  cap.

All reproducible Grok must-fixes were written red-first and fixed. Exact fixed
topology and premature-completion checks were added beyond the minimum finding.
A separate local actual-file reviewer found no remaining ship blocker. Treat
this as one external advisory plus one local review, not a synthesized
multi-lineage verdict.

- Executable Node client: **19/19**.
- Targeted Python WebSocket/server: **27/27** (22 WebSocket + 5 launch).
- Red-first evidence includes missing replay, user-visible seed propagation,
  pre-completion disconnect paint fencing, zero/non-fixed expected topology,
  malformed fixed chunks, premature completion, oversized chunks, bounded
  replay, counter-neutral looping, and replay-only decode degradation.

## State at handoff

- [[State]] is the current truth.
- Exact final HTML/CSS/JavaScript assets are synchronized. The initial clean
  relaunch refused exact static-evidence drift from 62 generated `.pyc` files
  totaling 1,006,933 bytes across aiohttp and dependencies. There was no
  package/source version drift. After exact path/count/byte validation, only
  those caches were removed and the full frozen evidence returned to exact lock
  equality. The no-bytecode wrapper relaunched a clean instance and the exact
  replay build passed real-browser reacceptance at 81 frames / 21 chunks / 21
  ACKs, complete server state, active replay, changing screenshots with stable
  counters, and zero console errors.
- The dedicated pod, worker, and local tunnel remain intentionally running for
  the user's follow-up test. No transient provider endpoint is persisted here.
- No secret value or `.env` content entered review, evidence, or the brain.
- No exact provider-billed total is known; none is inferred.
- No git commit was created.

## Next moves

1. Let the user compare the accepted detailed default with the terse prompt while
   describing the prompt-conditioned limitation honestly.
2. When the user is finished, close server/tunnel/worker gracefully, prove reap
   and zero frozen-runtime-tree drift, then stop the pod.
3. Resume the qualified evaluator, Round-A/Round-B, compatible long-horizon,
   and performance-authorized gates. Do not promote these diagnostics into a
   quality, FPS, TTFF, or sustained claim.

## See also

- [[Handoffs]]
- [[State]]
- [[CF1-Prompt-Conditioned-Motion]]
- [[CF1-H100-Runtime-Preflight]]
- [[Pegasus-1.5-Video-Judge-Calibration]]
- [[Browser-Streaming-Transport]]
- Prior: [[session-17-frozen-h100-generation-and-browser-acceptance]]
