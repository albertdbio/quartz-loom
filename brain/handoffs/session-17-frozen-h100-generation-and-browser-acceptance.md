---
type: handoff
status: active
session: 17
date: 2026-07-20
description: "Froze the H100 runtime, completed exact CUDA smoke/artifact/upload and persistent-worker lifecycle acceptance, then proved genuine CF++1 browser paint and 21 presentation ACKs; localhost remains live for the user's manual test."
branch: main
key_commits: []
prior_handoff: "session-16-acceptance-evidence-hardening"
---

# Session 17 Handoff — frozen H100 generation and browser acceptance

## TL;DR

- The schema-v2 H100 runtime/evidence lock is frozen and the unchanged
  authorizer passed. See [[CF1-H100-Runtime-Preflight]] and
  [[Pinned-CF1-CUDA-Bootstrap]].
- The exact 45-forward CF++1 path completed one-block and full
  21-block/81-PNG H100 smokes, deterministic H.264 assembly/full decode, and
  one development-only exact-byte upload to Gemini plus TwelveLabs. Provider
  judgments disagree and remain non-gating; see [[CF1-Latent-Pull-and-Smoke]]
  and [[CF1-Development-Video-Artifact]].
- Persistent-worker manifest SHA
  `30bfe91b4742bbd7e04966f9baed75562fce3c0a501b05c0063a45b7c54de115`
  is accepted: two distinct seeds reused one PID/instance for 21 chunks/81 PNGs
  each, then identity-fenced forced death proved backend/registry poison and
  worker reap. See [[CF1-Persistent-CUDA-Worker]].
- The genuine opt-in CF++1 H100/frozen-runtime page completed 81 painted frames
  in 21 chunks with 21 post-paint ACKs, server completion, a visible generated-
  fox canvas, and zero console errors. See [[Browser-Streaming-Transport]].
- The first acceptance instance shut down cleanly. The user then explicitly
  requested the URL, so a fresh warm instance remains intentionally live at
  `http://127.0.0.1:8765/`; keep the pod RUNNING until they finish.

## Frozen execution evidence

- Runtime evidence SHA: `8209043b4ebecc85f0e844f9c040b54fc1685104fe9e0b361ce9ee6d060b0c6c`.
- Frozen runtime-lock SHA: `d4d163d635ecbafb5b11bbe54cca7bdd5e9f80c1edc23ed96972821940ecc692`.
- Locked runtime/native/bound-probe identities: `858c1600…`, `8884c4be…`,
  and `070efce1…`.
- Current stack identity: `349c79f42726697310270a4e57395693cd16bab0ae7c903fd692dcfe995d5404`.
- Accepted worker identity: `59127c702b5c7cd394e7f9921e22d09b4ce05d2e4ef14dbf2d4b56ab64228fc8`.
- Full-smoke manifest SHA: `d363fbe1b65b167a5cf731aa194b00676f23a4d6cdcdaa0f89cfe8a7c960ba99`.
- MP4 SHA / artifact-manifest SHA: `8c892d63…` / `1b33d989…`.
- Dual-provider development-understanding manifest SHA: `e5f48cdf…`.

Worker warm took 517.196 seconds. The manifest's serial PNG-readiness timings
are diagnostics and explicitly authorize no performance claim.

The initial real run generated 1,649 bytecode files inside the runtime tree.
They were removed only after exact count/hash identification; the authorizer
was replayed and the full smoke subsequently changed zero environment files.

## Real browser evidence

The accepted request used the exact fox prompt, SHA-256
`fcaa04670647e417d16a4562d1027ca97a04ebcf9bed32c201fedd62ed9f785a`.
The page text identified the opt-in CF++1 H100 backend and frozen runtime rather
than the fake backend.

- DOM: `streamState=complete`, `renderedFrames=81`, `renderedChunks=21`,
  `ackCount=21`, `expectedFrames=81`, `serverCompleted=true`.
- Exact status: `Complete: 81 frames painted in 21 chunks; 21 presentation ACKs sent.`
- Canvas: present and visible; intrinsic 832×480, client 832×468,
  `display:inline`, `visibility:visible`, opacity 1, aria-label
  `latest decoded frame`.
- The captured browser view visibly showed the generated fox frame.
- Browser console errors: `[]`.

This proves one bounded localhost execution-to-presentation chain. It does not
authorize a quality, FPS, TTFF, P95, sustained, or public-deployment claim.

## Review and verification

- Focused browser/worker/acceptance/security tests: **72/72**.
- Affected worker/process/acceptance tests: **80/80**.
- Full discovery on Python 3.12.9: **434/435**. The sole failure is unrelated
  and preexisting: the recursive-JSON test expects `invalid_json` for 1,100
  nested arrays, while this Python parses them and returns `invalid_command`.
- Diff-fed consensus parent `ses_07eba6ca6ffeZao49EpwVikquA` timed out. Kimi
  and Grok child finals were both GO for the bounded worker-then-browser
  sequence; Fable produced zero tokens, no rate-limit signal occurred, and no
  automatic Opus fallback ran. Completed-voter cost was **$0.417366**.
- No git commit was created and no secret value entered review or evidence.

## State at session close

- [[State]] is the live truth. RunPod MCP control works.
- The original acceptance server/worker were gracefully removed.
- A newly warmed manual-test instance is intentionally live at
  `http://127.0.0.1:8765/`; do not stop it until the user is done.
- RunPod billed spend for this sequence is still unknown; do not invent it.
- No provider SSH endpoint or credential is persisted here.

## Likely next moves

1. Let the user exercise the localhost URL without replacing the live surface.
2. When they confirm completion, gracefully close server/tunnel/worker, verify
   process reap and zero runtime-tree drift, then stop the pod.
3. Resume evaluator qualification, complete Round-A/Round-B evidence, and the
   compatible long-horizon policy/cache decision.
4. Only after those gates and project-level GO, collect performance-authorized
   ≥60-second exact-stack evidence.

## See also

- [[Handoffs]]
- [[State]]
- [[CF1-Persistent-CUDA-Worker]]
- [[Browser-Streaming-Transport]]
- Prior: [[session-16-acceptance-evidence-hardening]]
