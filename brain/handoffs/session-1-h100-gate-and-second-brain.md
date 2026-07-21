---
type: handoff
status: archived
session: 1
date: 2026-07-19
description: "Measured CF++ 1/2-step H100 paths, corrected rolling TAEHV to 81 frames, achieved 31.04 warm e2e fps, failed the 7/10 quality gate, recorded unanimous Verdict B, stopped the pod, and created the Codegraph-backed second brain."
branch: main
key_commits: []
prior_handoff: ""
---

# Session 1 Handoff — H100 gate and second brain

## TL;DR

- CF++ 1-step plus corrected rolling TAEHV achieved 31.04 fps warm e2e,
  a 0.242s first post-decode GPU-RGB CUDA event, and a 37.62ms derived
  effective-frame P95 on one H100 at 480×832. Later audit clarified that the
  historical names overclaimed: neither CPU payload readiness nor browser
  visibility was measured.
- Blind full-video quality was 5.67/10, below the 7/10 gate; unanimous consensus
  chose Verdict B per [[ADR-001-quality-qualified-headline]].
- Fixed the rolling TAEHV startup trim invariant and restored 73-frame outputs
  to 81 frames; see [[Gotcha-Rolling-TAEHV-Context-Trim]].
- Updated PLAN, article, quality report, and executable roofline; stopped the
  RunPod pod at an estimated cumulative spend of $11.69.
- Added this Codegraph-backed brain and made `[[State]]` the next session's
  starting point.

## What this session worked on

- **H100 performance matrix** — measured CF++ 1-step and 2-step with full Wan
  VAE, batch TAEHV, serial rolling TAEHV, and overlapped rolling TAEHV.
- **Streaming correctness** — corrected decoder context trimming, archived the
  logs/metrics, and verified every audited MP4 has 81 frames.
- **Quality gate** — expanded to three prompts and nine blinded comparison
  videos; CF1 won two prompts but missed the absolute quality bar.
- **Decision review** — gpt-5.6-sol, kimi-k3, and grok independently reviewed
  the evidence and unanimously selected Verdict B.
- **Durable memory** — introduced `brain/`, note conventions, handoff workflow,
  and a local `.codegraph/` index backed by the custom memory layer.

## Decisions made

- [[ADR-001-quality-qualified-headline]] — performance and quality are one
  contract; 31.04 fps alone does not close the public target.

## New gotchas

- [[Gotcha-Rolling-TAEHV-Context-Trim]] — latent context must be converted to
  RGB-frame trim dynamically; a fixed 12-frame trim corrupts startup length.

## State at session close

[[State]] holds the live truth. The performance milestone is closed; the
quality-qualified headline, broad audit, and 60-second sustained run are open.
No GPU pod is running and no git commit was created.

## Verification evidence

- `bench/results/h100_cf1_taehv_overlap_metrics.json` — measured 31.0394 fps.
- `bench/results/h100_quality_eval.json` — CF1 5.67, SF4 4.67, CF2 4.33.
- All 14 audited MP4s were decoded and counted at exactly 81 frames.
- `python3 -m py_compile roofline/roofline.py` passed; executable report asserts
  the measured gate, Verdict B, stale-headline removal, and 9/9 anchors.
- `scripts/codegraph-local status .` reports 13 files, 82 nodes, 191 edges,
  seven memory notes, WAL, and an up-to-date index. A live
  `codegraph_explore` query returned both anchored notes beside
  `report_measured_h100_gate`.
- The index has 23 resolved memory edges and zero dangling memory anchors.
  Exact note-path explore queries can misrank into Python on v1.4.0; the CLI
  note query and anchored-memory path work, and AGENTS.md records the workaround.
- Python LSP diagnostics were clean. Markdown has no configured LSP; JSON's
  Biome LSP is unavailable because installation was previously declined.

## Likely next moves

- Read `[[State]]`, `PLAN.md`, and [[ADR-001-quality-qualified-headline]].
- Freeze CF++ 1-step plus TAEHV and pre-register the quality-repair sweep.
- Run at least 10 stratified prompts with multiple judge families and human
  review; preserve full videos and exact frame-count assertions.
- Prove a sustained run of at least 60 seconds before revisiting a headline.
- Consider static KV-ring work only after quality evidence identifies it as the
  best use of GPU budget. These moves are provisional; new evidence may pivot.

## See also

- [[Handoffs]]
- [[State]]
