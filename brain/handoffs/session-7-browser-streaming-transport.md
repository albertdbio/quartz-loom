---
type: handoff
status: active
session: 7
date: 2026-07-20
description: "Verified the fourth Pegasus upload, hardened NDJSON and same-origin binary WebSocket delivery, and exercised 81-frame presentation plus replacement in a real browser."
branch: main
key_commits: []
prior_handoff: "session-6-streaming-service-boundary"
---

# Session 7 Handoff — Browser streaming transport

## TL;DR

- The adjusted TwelveLabs permission works: a fourth bounded real Pegasus 1.5
  upload completed through the strict parser. It reinforces transport viability
  and evaluator inflation, not calibration readiness.
- [[Streaming-Service-Boundary]] now extends through hardened loopback NDJSON
  and same-origin binary WebSocket adapters to a real browser presentation ACK.
- [[Browser-Streaming-Transport]] records 81 frames/21 chunks/21 ACKs in about
  5.1 seconds and a successful same-socket replacement after five frames. The
  visible backend remains fake PNG.
- Consensus session recovery found real replacement/commit races. Each failed
  red before its fix; [[Gotcha-Transport-Write-Is-Not-Presentation]] preserves
  the cross-adapter lesson.
- The real persistent process worker, corrected CUDA generator, provenance
  result, evaluator qualification, and sustained exact-stack evidence remain
  open. The project-level gate stays NO-GO.

## What changed

- Added `bench/streaming_websocket.py`, `bench/streaming_web_demo.py`, external
  CSP-safe browser assets, pinned Python-3.9-compatible aiohttp dependency, and
  executable Python/Node tests.
- Hardened the WebSocket surface with exact Host/Origin/subprotocol checks,
  one-time session nonces, binary rasters, a two-chunk presentation window,
  post-binary commit tokens, lifetime ID uniqueness, bounded sends, sanitized
  failures, and replacement-safe retirement.
- Hardened the NDJSON sibling for socket write/drain races, monotonic ACK
  clocks, cancellation ordering, no-ACK recovery, lifetime job IDs, and the
  same replacement-safe stale-send interpretation.
- Tightened CPU preflight empty checkpoint sets, raster signature truth, and
  initial-clock terminal failure handling.

## Review and evidence

- Required consensus aliases were `claude-opus,kimi-k3,grok`; the outer browser
  review timed out after 600 seconds. Parent
  `ses_0818704d9ffeNs36KPw31kgHEy` retained one completed Grok analysis while
  Fable/Opus and Kimi had no final.
- Grok found stale-retirement disconnect, commit-after-retire, and stale-client
  commit races. All reproduced red. A lifetime-delivery-ID regression and an
  executable client presentation-order test were added at the same time.
- Searching the sibling adapter found the same stale-send disconnect bug in
  NDJSON; it also reproduced red before the fix.
- The smaller post-fix panel also timed out, but recovery from
  `ses_08178b6e8ffezbHmo49X5yKZKd` retained Fable and Grok finals. Both judged
  all five defects fixed for the fake milestone; Kimi again had no final.
  Fable's missing replacement-before-commit coverage is now an executable Node
  test. Grok's remaining P1 was rejected after reading the on-disk lines omitted
  from its excerpt: cleanup, retirement, notification, and re-raise already
  exist.

## Verification

- Complete Python 3.9 suite: **182/182**.
- Focused service + NDJSON + WebSocket: **77/77**.
- Executable Node client: **4/4**; JavaScript syntax and Python compilation pass.
- Real in-app browser: **81/21/21 in 5.092s**, zero warnings/errors.
- In-flight same-socket replacement after five frames: replacement **81/21/21
  in 5.154s**, zero warnings/errors.
- Fourth Pegasus upload: official v1.3 endpoint, `pegasus1.5`, clean stop,
  scores 9/9/9/8/9, strict parser pass. Four-call estimate is ~$0.0150.
- No GPU was started, no secret value was printed/persisted, and no commit was
  created.

## Likely next moves

1. Build the persistent fake process-worker protocol red-first: capacity-one
   pull IPC, worker/job/chunk epoch fencing, bounded lengths, kill/reap/poison,
   environment allowlist, immutable payloads, and typed terminal provenance.
2. Do not wire CUDA until the absent upstream checkout, configs, generator
   checkpoint, and TAEHV weights are restored and strictly pinned.
3. Continue the preregistered human calibration and long-horizon policy work in
   parallel; browser transport does not change evaluator or headline gates.
4. Recover timed-out consensus children before re-running panels.

## See also

- [[Handoffs]]
- Prior: [[session-6-streaming-service-boundary]]
