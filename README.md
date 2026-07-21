# realtime-video

Research + product monorepo for real-time video generation and editing on a single GPU.

**The story so far** (full write-up with embedded results: [`blog/rtv-motion-collapse.html`](blog/rtv-motion-collapse.html)):
a 1-step causal video-diffusion student that renders beautifully but refuses to move
objects. We localized the failure to the Stage-3 distillation objective (not the
decoder, not the init, not our port — verified against the reference implementation's
own released checkpoints), replaced it with a score-gradient-matching objective,
normalized its Fisher weighting, swept λ, and measured the motion×coherence frontier
move from "perfect stills" (0.29, 10) to "real transport at watchable coherence"
(2.65, 7.0) — zero architecture changes.

## Layout
- `studio/` — SolidStart web app: real-time video editing from your camera **or** a
  voice-driven "story canvas" that dreams scenes as you narrate. Three modes:
  hosted API, self-hosted open model, story (either engine). Deployable to Vercel.
- `openstudio-server/` — self-hosted realtime restyle server (StreamDiffusion
  sd-turbo, ~10-14 fps on a rented 4090/L40S), websocket wire contract v1,
  quality bench (`bench_quality.py`) that picked the shipped config.
- `bench/` — the measurement system: calibrated displacement (motion) and coherence
  metrics, batch scorer, and the training-trajectory results.
- `research/` — findings, experiment plans, per-run verdicts, score data, and the
  vendored SGMD delta for Causal-Forcing (`research/sgmd-causal-forcing/`).
- `brain/`, `PLAN.md` — living project state, decisions, and the honest-measurement
  rules this repo runs on (a green build is the floor, not the finish line).

## Quick start (studio)
```bash
cd studio && pnpm install
cp .env.example .env   # add your DECART_API_KEY
pnpm dev               # http://localhost:3000
```
Self-hosted mode needs `openstudio-server/` running on a GPU box — see its README.

## Agent workflow

Large parts of the measurement system and the experimental audits were produced by
long-running **Codex (GPT-5.6) sessions operating autonomously** — pointed at a goal
with hard constraints, then left to work for hours at a time. They were relentless
workhorses:

- ran the **P0 root-cause audit** end-to-end on a rented H100 (provisioning, restoring
  a pinned training stack, generating and blind-scoring the evidence, writing the ADR)
  for about $5 in compute;
- built the **calibrated displacement and coherence metrics** (`bench/`) red-first —
  writing failing synthetic tests before implementations, then adversarially reviewing
  their own code, filing prioritized defect lists against themselves, and fixing them;
- **batch-scored every training corpus** with exact reproducibility — independent
  spot-checks matched their reported numbers byte-for-byte, 11/11;
- reported **honest nulls** rather than grinding for favorable numbers (skipped cells
  are marked, degraded clips are flagged, absent baselines are stated as absent).

The sessions routinely consumed 800k+ tokens each and kept working through
environment failures and restarts. The pattern that worked: one bounded deliverable
per session, evidence-first constraints, and a standing rule that a green test suite
is the floor, not the finish line.
