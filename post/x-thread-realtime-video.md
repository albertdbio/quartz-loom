# X thread — "push realtime video" launch (v2 — post-consensus, ready to post)

> Status: consensus-reviewed 2026-07-22 — CLI panel `claude-opus,kimi,grok`
> (concrete: claude-opus-4-8, kimi-k2.6, grok-4.5), 3/3 GO-WITH-FIXES; all
> blocking + should-fix edits applied below. Attachments: promo cut
> (`post/promo/realtime-cut.mp4`), YouTube demo link, blog link.

---

**1/**
Real-time video generation is the most fun I've had with AI.

I built a studio where you point your camera, describe an edit, and watch
yourself transform — live.

Anime. Claymation. Oil painting. While you move.

Two engines under the hood — a hosted API and open weights. Demo below 👇

[attach: promo cut video]

**2/**
Three modes in the app:

- **Hosted mode** — realtime restyle through @DecartAI's API (~24fps, their
  model, their quality — attributed because it's excellent). That's the smooth
  footage in the clip.
- **Self-hosted mode** — open-weights (sd-turbo + StreamDiffusion) on one
  rented 4090: ~10 fps, ~150ms p50 glass-to-glass, restyle config picked by
  our quality bench. Interactive, not yet real-time — and fully yours.
- **Story mode** — no camera. You narrate, the canvas dreams the scene.

**3/**
Under the app is the research lane: CHASING quality-qualified real-time
generation on one GPU.

Our 1-step student (Causal-Forcing++ distill of Wan-1.3B) runs 480×832 at
~27–31 fps warm end-to-end on a single H100 — at ~2.5/10 blind motion
quality.

That number pair IS the story. Throughput was never the hard part.

**4/**
The hard part: the model renders beautifully and refuses to MOVE anything.

"A yellow car drives across the frame" → a perfect car, rocking in place.

We call it motion collapse. Identity ✓. Displacement ✗.

**5/**
A $5 root-cause audit on one H100 — hold the latents fixed, swap one stage at
a time, blind-score the motion — cleared the decoder and the history cache:
the stillness is baked into the latents.

A second experiment pinned the culprit: the distillation objective itself.
Reverse-KL is mode-seeking, and the stillest mode always wins.

**6/**
The fix: swap in a score-gradient-matching objective, normalize its Fisher
weighting, sweep λ. Zero architecture changes.

The frontier moved from (motion 0.29, coherence 10) — perfect stills — to
(motion 2.65, coherence 7.0) — real transport at watchable coherence.

(That 7.0 is the coherence axis, not our 7/10 motion-quality gate — that gate
is still open, and we say so.)

Full write-up with embedded results: [blog link]

**7/**
Rules this project runs on:

- quality reported next to EVERY fps number
- nothing under 24 fps gets called "real-time"
- warm/cold separated, every percentile labeled

The measurement system (calibrated displacement + coherence metrics) was built
red-first by autonomous Codex sessions running for hours at a time.

**8/**
Full 3-min demo: [YouTube link]

This is the road to world models you can stand inside. We're at mile one —
but the car finally drives. 🎥⚡

---

## Video captions (final, post-consensus)

1. "I built a live AI video studio"
2. "describe an edit - watch it apply live"
3. "claymation. one click. (hosted mode - Decart API)"
4. "open-weights model, one GPU - ~10 fps, interactive"
5. "story mode - you narrate, the canvas dreams the scene"
6. End card: "real-time video research" / "full demo + write-up in thread"

## Claims map (verified against artifacts, consensus-adjudicated)

| Claim | Source |
|---|---|
| Hosted ~24fps via Decart API, attributed | Submission demo; Decart's service |
| Self-hosted ~10 fps / ~150ms p50 / bench-picked config, "interactive" | PLAN ledger 2026-07-21 (est_fps 10.5, e2e p50 152ms) |
| ~27–31 fps warm e2e + ~2.5/10 blind motion, paired | blog header + §02 |
| $5 audit → latents; separate experiment → objective | blog §04, §07 |
| Frontier (0.29,10)→(2.65,7.0), axes named, gate disclaimed | README; batch_scores_2axis |
| World models line is directional ("road to", "mile one") | aspiration, not achievement |
