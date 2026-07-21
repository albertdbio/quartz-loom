---
type: decision
status: accepted
date: 2026-07-19
description: "A real-time headline requires throughput and quality from one exact, byte-bound stack. Archived rolling-TAEHV performance and full-batch-TAEHV quality are separate development signals, so the replacement gate remains open."
anchors: ["roofline/roofline.py#report_measured_h100_gate", "bench/quality_sweep.py#evaluate_gate", "PLAN.md", "post/rooflines-for-realtime-video-diffusion.md"]
related: ["[[Gotcha-Rolling-TAEHV-Context-Trim]]", "[[Gotcha-Seed-Is-Not-Artifact-Provenance]]", "[[Gotcha-Quality-Decoder-Must-Match-Performance-Path]]", "[[State]]"]
---

# ADR-001 — Performance does not qualify the headline by itself

## Decision

Every throughput result is reported beside its quality result. The public
quality-qualified target is reached only when one configuration satisfies all
of the following:

1. At least 29 fps warm end-to-end at 480×832 on one H100.
2. At least 7/10 absolute quality on a pre-registered, stratified suite of at
   least 10 prompts.
3. Multiple independent judge families plus human review.
4. At least 60 seconds of sustained execution.
5. An apples-to-apples protocol before any “beats MotionStream” language.

## Current application

CF++ 1-step plus corrected rolling TAEHV measured 31.04 fps warm e2e and a
0.242s first post-decode GPU-RGB CUDA event, but scored 5.67/10 in the initial
blind quality audit. The historical runner named that field
`first_visible_rgb_s`; it did not time CPU payload readiness, transport, or
browser presentation.
The fixed voter set (gpt-5.6-sol, kimi-k3, grok) unanimously selected Verdict B:
performance milestone achieved, headline open.

The article and executable roofline may describe the numerical 7.03% margin
over MotionStream's published 29 fps only with the protocol caveat and failed
quality gate. They may not claim system superiority.

## Why

Throughput can be inflated by a student or decoder that removes useful temporal
detail. Quality is therefore part of the performance contract, not a later
marketing check. The split Verdict B preserves the useful systems result while
preventing a benchmark win from laundering a failed visual result.

## Evidence-status clarification — 2026-07-19

The decision above remains accepted, but later provenance recovery narrowed
what its historical evidence can establish. The archived 31.0393506-fps result
used corrected rolling three-latent TAEHV. The three-prompt 5.6667/10 quality
suite used one full-batch TAEHV decode. Their prompt/seed labels match, but
their encoded outputs do not; seed equality is not artifact provenance.

Accordingly, those two numbers are separate archived development signals, not
one exact-stack gate result. The 31.04-fps arithmetic remains useful and the
5.67 score still motivates quality repair, but neither closes the replacement
protocol. A new result must bind source/config/runner/checkpoint/decoder,
initial noise, latent, encoded-media hashes, complete blind rating evidence,
and the long-horizon performance/media tensor before this ADR can qualify a
headline.

## Replacement-protocol status — 2026-07-19

The owner-directed pre-hardening panel unanimously ruled NO-GO for freezing the
replacement protocol, starting exact-stack GPU generation, or using it to
qualify a headline. A post-hardening retry timed out after 600 seconds without a
completed result; a timeout is not a vote and does not lift the freeze.

At that review point, local P0 work remained at the evidence boundary: the gate
had to recompute quality
aggregates from normalized ratings, derive performance from bound raw timing
events, and rerun media inspection against the actual artifacts. Only a later
completed GO review of the actual revised files can authorize protocol freeze.

## Evidence-boundary update — 2026-07-19

The three implementation P0 boundaries named above are now closed locally:
gate evaluation recomputes the full aggregate from normalized ratings and the
operator blinding secret, re-audits physical media bytes and ffprobe metadata,
and derives performance metrics from raw timestamps bound to selected audited
artifacts. Adversarial regressions are part of the passing contract suite.

This does not change the accepted decision or authorize GPU work. Deterministic
finalist-selection report, ranking, lock, and gate-recomputation machinery is
implemented and tested, but no real complete Round-A/Round-B development
evidence has been collected and therefore no evidence-backed finalist exists.
Source, config, checkpoint, decoder, runner, and independent-model pins remain
open; both recovered runners are historical-only; and no real confirmatory
tensor exists. The timed-out review also retained two child NO-GO finals
(Fable-5 and Grok-4.5) but no Kimi final, which is partial review signal rather
than a completed consensus verdict. See
[[Gotcha-Consensus-Timeouts-Preserve-Voter-Sessions]].
