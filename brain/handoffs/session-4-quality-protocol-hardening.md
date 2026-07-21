---
type: handoff
status: archived
session: 4
date: 2026-07-19
description: "Separated legacy performance/quality evidence, completed local gate and deterministic selection machinery, verified Pegasus 1.5 transport but failed its judge calibration, and kept GPU generation frozen pending qualified evidence and review."
branch: main
key_commits: []
prior_handoff: "session-3-unsloth-puzzles-study"
---

# Session 4 Handoff — Exact-stack quality protocol hardening

## TL;DR

- Recovered the decisive provenance gap: archived 31.039-fps evidence used
  rolling three-latent TAEHV, while the legacy 5.67/10 suite used full-batch
  TAEHV. They are not one exact-stack gate result; see
  [[Gotcha-Quality-Decoder-Must-Match-Performance-Path]].
- Added a draft local quality-repair protocol, validator/gate implementation,
  schemas, CLI, and contract tests that bind manifests, blinding, ratings,
  media, and performance evidence.
- A completed panel requested `claude-opus`, `kimi-k3`, and `grok`; concrete
  `claude-fable-5`, `moonshotai/kimi-k3`, and `grok-4.5` unanimously returned
  NO-GO for freeze, GPU generation, and headline use.
- The revised-artifact consensus bridge timed out after 600 seconds. Direct
  session recovery found completed NO-GO analyses from concrete
  `claude-fable-5` and `grok-4.5`; the Kimi-K3 child produced no final. This is
  partial review signal, not a completed three-voter verdict.
- Gate hardening now recomputes the complete rating aggregate, re-audits media
  bytes/ffprobe output at decision time, and derives/binds performance from raw
  timestamps and selected audited artifacts. The focused local contract suite
  passes.
- Deterministic Round-A/Round-B finalist-selection implementation and tests are
  complete: the gate recomputes the ranked winner and validates its lock from
  the underlying development inputs. No real Round-A/Round-B evidence has been
  collected, so no evidence-backed finalist is locked.
- Pegasus 1.5 now has a strict synchronous structured adapter, CLI, evidence
  schema, and captured scrubbed fixture. Three real fox canaries passed
  transport/parser checks but contradicted the archived Gemini ordering, so
  [[Pegasus-1.5-Video-Judge-Calibration]] records transport GO / calibration
  NO-GO. The canaries are not gate evidence.
- No GPU was restarted; only ~$0.0114 of API calibration spend was added, and
  no git commit was created.

## What this session worked on

- **Evidence recovery** — traced archived runner, manifest, metrics, and media
  artifacts far enough to separate rolling performance from batch quality and
  record [[Gotcha-Seed-Is-Not-Artifact-Provenance]].
- **Replacement contract** — built the local `quality-repair-v1` protocol path
  around pinned provenance, exact prompt/seed/system grids, secret-keyed
  blinding, complete ratings, aggregate/media/performance reports, and a
  fail-closed gate. The gate no longer trusts rehashed aggregate, media, or
  performance summary envelopes as authoritative evidence.
- **Finalist selection** — implemented and tested complete Round-A/Round-B
  manifest/rating validation, raw-timing-bound development performance,
  deterministic ranking, selection-report/lock construction, and gate-time
  winner recomputation.
- **Independent review** — supplied the actual scrubbed artifacts to the
  owner-directed panel and preserved both its unanimous NO-GO and the later
  timeout as distinct outcomes under [[ADR-002-consensus-review-panel]].
- **Claim correction** — updated [[ADR-001-quality-qualified-headline]],
  `PLAN.md`, and [[State]] so archived throughput and quality are no longer
  represented as a single qualified measurement.
- **Video-judge calibration** — integrated official `pegasus1.5` synchronous
  structured analysis behind a scrubbed, hash-bound evidence boundary. The
  API key stayed local and untracked; it was never printed or persisted.

## Decisions made

- [[ADR-001-quality-qualified-headline]] — exact-stack artifact identity is part
  of the throughput-plus-quality contract; neither archived signal closes it.
- [[ADR-002-consensus-review-panel]] — future panels request
  `claude-opus,kimi-k3,grok`; Fable is the first concrete Claude choice and the
  tool, not the caller, owns any Opus-4.8 rate-limit retry.

## New gotchas

- [[Gotcha-Seed-Is-Not-Artifact-Provenance]] — prompt and seed labels do not
  prove code/config/noise/latent/decoder/media identity.
- [[Gotcha-Quality-Decoder-Must-Match-Performance-Path]] — batch and rolling
  TAEHV outputs cannot qualify each other's results.

## State at session close

[[State]] remains the live truth. The replacement protocol is local and draft,
both recovered runners remain historical-only, and exact-stack generation is
frozen. The three implementation P0 boundaries are closed locally. Unresolved
source/config/checkpoint/model pins remain, no real complete Round-A/Round-B
development evidence or evidence-backed selection lock exists, real
confirmatory data does not exist, and no completed post-hardening panel has
cleared the work. Pegasus readiness is `calibration-failed`, and protocol freeze
now requires every model family to be `quality-qualified`. The deterministic
finalist-selection implementation and tests are complete; evidence collection
has not begun.

## Verification evidence

- All 14 recovered MP4s were audited as 832×480 H.264/yuv420p at 16 fps with
  exactly 81 decoded frames; they remain legacy/development media.
- The completed pre-hardening panel returned all three requested lineages as
  concrete Fable-5, Kimi-K3, and Grok-4.5 voters with a unanimous NO-GO.
- The timed-out session `ses_083124e38ffeGJLxe1Jh7m0v9Z` retained two child
  finals: Fable-5 and Grok-4.5 both NO-GO. Kimi-K3 had zero output/no final, so
  the recovered result remains incomplete. See
  [[Gotcha-Consensus-Timeouts-Preserve-Voter-Sessions]].
- The focused quality contract/CLI/sweep test suite passes; Python compilation,
  every schema's JSON parse, and draft protocol validation also pass. Forced
  frozen validation correctly rejects the draft.
- The expanded suite now passes 69/69 after binding each actual raw-evidence
  report through normalized rows, aggregate summaries, development selection,
  and the final gate. Pegasus responses are re-parsed before acceptance;
  missing, reordered, duplicate, swapped, or score-divergent evidence fails.
  Unbound historical aggregation requires an explicit legacy opt-in.
- Round-A/Round-B finalist-selection report, ranking, lock, and gate-recompute
  paths have deterministic contract coverage. This verifies the machinery only;
  no real development-selection evidence was produced.
- Focused `../oh-my-openagent` consensus tests pass 55/55, covering
  Fable-first resolution plus the one-time Opus-4.8 retry for detected Fable
  rate limits.
- The post-hardening `claude-opus,kimi-k3,grok` call timed out after 600 seconds
  in `ses_082c43944ffevDHbs620lDSV3t`. Recovered children were Fable-5
  GO-WITH-FIXES on the library chain, Grok-4.5 project NO-GO, and Kimi-K3 with
  zero output. Fable completed directly, so Opus fallback was not triggered.
  This remains partial review signal; it does not satisfy the completed-GO gate.
- Three real 5.0625-second fox canaries returned clean Pegasus responses and
  strict-parser passes: CF1 9.0, CF2 8.6, SF4 8.0. Archived Gemini P0 instead
  ranked SF4 6, CF1 5, CF2 4, so the calibration failed despite working
  transport. Estimated cost was ~$0.0114 for 15.1875 video seconds and 534
  output tokens.
- The RunPod pod stayed stopped. GPU spend remains approximately $11.69; only
  the ~$0.0114 API estimate was added, and no git commit was created.

## Likely next moves

- Collect the registered Round-A/Round-B manifests, ratings, and raw-timing-bound
  development performance needed to produce an evidence-backed selection report
  and finalist lock through the now-tested deterministic path.
- Calibrate Pegasus on a registered discriminative set against independent
  human/model judgments, or replace it; do not fan out canaries or gate media
  until the family is `quality-qualified`.
- Resolve every CF/SF runner, source-diff, config, checkpoint, decoder, and
  independent-model pin; keep historical runners ineligible for confirmation.
- Rerun diff-fed consensus with `claude-opus,kimi-k3,grok` over the actual final
  files; if the bridge times out, recover each child session and still require a
  complete panel before GO. Only a completed GO may freeze the protocol or
  authorize GPU work.
- After GO, generate the exact confirmatory tensor and pair its quality evidence
  with a ≥60-second sustained performance run. These moves are provisional.

## See also

- [[Handoffs]]
- [[Pegasus-1.5-Video-Judge-Calibration]]
- [[State]]
- Prior: [[session-3-unsloth-puzzles-study]]
