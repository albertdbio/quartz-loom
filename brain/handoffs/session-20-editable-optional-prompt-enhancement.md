---
type: handoff
status: active
session: 20
date: 2026-07-20
description: "Shipped one user-owned editable prompt, exact-text Generate semantics, explicit optional Gemini 3.1 Flash-Lite enhancement, race-safe provenance, and live 81/21/21 browser acceptance with replay left visible at loopback."
branch: main
key_commits: []
prior_handoff: "session-19-live-temporal-prompt-jpeg-release"
---

# Session 20 Handoff — editable optional prompt enhancement

## TL;DR

- The browser now has one editable prompt textarea. `Generate` sends exactly
  its current text; no automatic enhancement or separate effective-prompt field
  stands between the user and generation.
- `Enhance prompt` is an explicit optional action. It uses
  `gemini-3.1-flash-lite` with transform
  `gemini-3.1-flash-lite-video-prompt-expansion-v2`, replaces the same field,
  and leaves the result editable before any start. See
  [[Browser-Streaming-Transport]].
- Prompt provenance and generation state are race-safe: edits invalidate only
  unused provenance, an already-sent start keeps its fence, late resolver
  results cannot overwrite an edit or an active request, and generation stays
  busy through completion.
- The client suite passes **29/29**. Live `a bouncing ball` enhancement returned
  in **1730 ms**; the reviewed prompt then completed 81 painted frames / 21
  chunks / 21 presentation ACKs, with replay left visible at loopback.

## What this session worked on

- **User-owned prompt path** — collapsed raw/effective inputs into one editable
  source of truth and made enhancement a reviewable, optional action.
- **Prompt lifecycle fences** — separated unused enhancement provenance from an
  already-sent start fence and retired stale resolver responses on edit/start.
- **Active-job and terminal-state UX** — held the busy state through terminal
  presentation and preserved specific protocol errors across socket close.
- **Live product acceptance** — exercised enhancement, genuine generation,
  browser paint/ACK, and replay on the user-visible loopback surface.

## Decisions made

- The prompt textarea, not the resolver, is the product source of truth.
  `Generate` sends exactly what the user can see and edit.
- Prompt enhancement is opt-in and disclosed. Its result replaces the same
  textarea and never starts generation automatically.
- Editing revokes only unused enhancement provenance. A start already sent
  retains its one-time provenance fence until accepted or retired.
- Generation remains busy through completion, and a terminal protocol error is
  more informative than the socket-close event that follows it.

## State at session close

- [[State]] is the live truth.
- The accepted replay remains visible on the loopback page for the user.
- This is a Serving/Product UI release and bounded browser acceptance, not a
  quality, throughput, or public-deployment claim.
- No git commit was created.

## Verification evidence

- Executable browser client suite: **29/29**.
- Actual-file standalone CLI consensus: Claude Opus, Kimi, and Grok each
  returned **SHIP**. The only conditional item was the static `/demo.js` alias,
  which the live accepted page demonstrably loaded.
- Live browser: `a bouncing ball` enhanced in **1730 ms** into the same editable
  textarea without starting generation.
- The subsequent request completed 81 painted frames / 21 chunks / 21
  presentation ACKs, and replay remained visible at loopback.

## Likely next moves

- Leave the accepted replay available while the user inspects the new prompt
  flow.
- Continue with the next user-visible Serving/Product motion or throughput
  improvement.
- Keep exact-text generation and optional enhancement as separate explicit
  actions as the UI evolves.

## See also

- [[Handoffs]]
- [[State]]
- [[Browser-Streaming-Transport]]
- [[CF1-Prompt-Conditioned-Motion]]
- Prior: [[session-19-live-temporal-prompt-jpeg-release]]
