---
type: state
status: active
date: 2026-07-21
description: "The motion-by-coherence instrument has now scored 96 clips across raw SGMD, normalized lambda=0.1, and normalized lambda sweeps. Lambda=0.2 owns the strongest observed late operating point (step 50: 1.693540 motion / 7.403891 coherence), contrary to the lower-lambda framing, with coverage and prompt-fidelity caveats explicit."
related: ["[[ADR-001-quality-qualified-headline]]", "[[ADR-002-consensus-review-panel]]", "[[ADR-003-serving-product-model-track-split]]", "[[Gotcha-Rolling-TAEHV-Context-Trim]]", "[[Gotcha-Seed-Is-Not-Artifact-Provenance]]", "[[Gotcha-Quality-Decoder-Must-Match-Performance-Path]]", "[[Gotcha-Consensus-Timeouts-Preserve-Voter-Sessions]]", "[[Gotcha-Async-Timeouts-Need-Task-Isolation]]", "[[Gotcha-Transport-Write-Is-Not-Presentation]]", "[[Unsloth-Puzzles-Systems-Study]]", "[[Pegasus-1.5-Video-Judge-Calibration]]", "[[Gemini-3.1-Video-Judge-Calibration]]", "[[Pinned-CF1-CUDA-Bootstrap]]", "[[CF1-H100-Runtime-Preflight]]", "[[CF1-Rolling-TAEHV-Session]]", "[[CF1-Latent-Pull-and-Smoke]]", "[[CF1-Development-Video-Artifact]]", "[[CF1-Persistent-CUDA-Worker]]", "[[CF1-Prompt-Conditioned-Motion]]", "[[Streaming-Service-Boundary]]", "[[Browser-Streaming-Transport]]", "[[Coherent-Displacement-Metric]]", "[[Displacement-Batch-Comparison]]", "[[Coherence-Metric]]", "[[SGMD-Training-Trajectories]]", "[[session-1-h100-gate-and-second-brain]]", "[[session-2-codex-codegraph-consensus]]", "[[session-3-unsloth-puzzles-study]]", "[[session-4-quality-protocol-hardening]]", "[[session-5-gemini-calibration-and-runner-preflight]]", "[[session-6-streaming-service-boundary]]", "[[session-7-browser-streaming-transport]]", "[[session-8-persistent-process-worker]]", "[[session-9-pinned-assets-and-cuda-bootstrap]]", "[[session-10-rolling-cuda-session-core]]", "[[session-11-runtime-preflight]]", "[[session-12-latent-pull-and-smoke]]", "[[session-13-development-video-artifact]]", "[[session-14-persistent-cuda-worker]]", "[[session-15-persistent-worker-acceptance]]", "[[session-16-acceptance-evidence-hardening]]", "[[session-17-frozen-h100-generation-and-browser-acceptance]]", "[[session-18-motion-diagnosis-and-looping-replay]]", "[[session-19-live-temporal-prompt-jpeg-release]]", "[[session-20-editable-optional-prompt-enhancement]]", "[[session-21-coherent-displacement-metric]]", "[[session-22-displacement-batch-comparison]]", "[[session-23-fork-grid-displacement-baseline]]", "[[session-24-sgmd-degraded-displacement]]", "[[session-25-openstudio-4090-open-model-server]]", "[[session-26-coherence-axis]]", "[[session-27-sgmd-training-trajectories]]"]
---

# Current State

## Mission and gate

Build a frontier-lab-grade portfolio around quality-qualified real-time video
generation: reproducible rooflines, benchmarks, systems work, and an interactive
demo. The hard real-time floor is 24 fps; the primary headline target is at
least 29 fps warm end-to-end at 480×832 on one H100 while passing an absolute
7/10 quality gate. Performance never appears without its quality result; see
[[ADR-001-quality-qualified-headline]].

The project has crossed the frozen-runtime, real CUDA generation, artifact,
persistent-worker lifecycle, and real-browser visibility/presentation
milestones. It has **not** crossed the replacement performance gate, the
quality-qualified evaluator gate, or the sustained headline gate.

The current role is bounded parallel measurement research. Strategy and latent-
generator experiment selection belong to the separate orchestrating researcher;
this track builds repeatable measurements and hands them off.

## What exists and is verified

- **The openstudio open-model serving path is live (serving/product track,
  [[ADR-003-serving-product-model-track-split]])** — `openstudio-server/`
  (sd-turbo StreamDiffusion, 512×512 2-step, TAESD, fp16, engine-agnostic
  binary WebSocket contract v1) is deployed on rented 4090 pod
  `52hu4efmx6bym0` and verified end-to-end: pod selfcheck PASS at
  **est_fps 10.5** (infer p50 70.27 ms; interactive, never to be called
  real-time), `contract_test.py` 12/12 over the real SSH tunnel, and a
  real-imagery probe at **e2e p50 152 ms** glass-to-glass with a
  prompt-conditioned watercolor restyle eyeballed. The white-box
  `test_server_protocol.py` (8/8) pins newest-frame-wins and prompt
  coalescing deterministically. The studio browser client (progress entry 32)
  is verified against the fake pipeline and awaits this pod. Pod deliberately
  RUNNING; see [[session-25-openstudio-4090-open-model-server]].
- **Coherent displacement is now measured locally** —
  `bench/displacement_metrics.py` scores exact 81-frame 480×832 MP4/PNG clips
  with CoTracker3 primary-object tracks, background-only homographies,
  screen/compensated two-space extent, adjacent-frame flow-warp residual, and
  DINOv2 median/p10 crop similarity. Raw flow magnitude cannot create credit.
  The red-first suite passes **14/14**; the official CoTracker3+DINOv2
  translating square scores **9.809/10** and displaced, while flicker and both
  camera counterexamples remain collapsed. On the P0 set, ten independent
  contents give Spearman **ρ=0.608, p=0.062**. D1 rolling versus D0 Wan differs
  by only 0.111 score and 0.00137 span/W; both are below the conservative 5.0
  decision threshold. See [[Coherent-Displacement-Metric]].
- **The scorer is now usable for experiment comparison** —
  `bench/displacement_batch.py` consumes direct condition/clip trees and emits
  full per-clip scorer JSON plus deterministic condition-by-prompt JSON and
  Markdown matrices. `--compare A B` means B-minus-A, missing cells remain
  explicit, and completed reruns prune stale tool-owned clip JSON. The real
  default-backend P0 regression reproduces all 11 session-21 fixture rows
  exactly; the example has 11 scored clips, nine holes, and ring ON-minus-OFF
  mean **-1.018855**. See [[Displacement-Batch-Comparison]].
- **The real fork grid is scored, with unavailable states kept honest** — the
  read-only 7-condition × 4-prompt grid produced 17 scored cells, seven
  explicit metric `ERROR` cells, zero holes, and four `INVALID`
  `oneforcing_1step` cells that were never scored. Filename-substring mapping
  pins the canonical prompt order and three comparisons share one backend pass.
  Vehicle exactly reproduces `cd_4step=1.232953`, `ode_4step=2.128855`, and
  `final_1step=0.286952`. Comparable-only means favor `cd_4step` over
  `final_1step` by **0.234997** and `ode_4step` over `final_1step` by
  **1.305990**; ODE exceeds CD by **0.310139**. This supports the native
  four-step Stage-2 versus final-one-step baseline, not the stronger claim that
  every pre-DMD condition beats both finals: final-two-step ball/barrel score
  high on transient recenter/drift rather than the requested sustained action.
  See [[Displacement-Batch-Comparison]].
- **Camera-compensation loss is explicit rather than silent** — only the
  dedicated insufficient-background-track condition can fall back to the
  existing screen-space measurement path. Such reports carry
  `coherence_degraded=true`, `camera_compensated=false`, a reason, and null
  camera-space diagnostics; compensated reports remain byte-identical. Batch
  cells mark fallback scores with `*`, exclude them from comparisons by
  default, and require `--include-degraded` for diagnostic comparison. Metric
  errors remain explicit `ERROR` rows; unexpected I/O/program failures abort
  without replacing completed output. The fast batch suite is **25/25**, and
  the learned P0 replay remains exact across all 11 clips.
- **The SGMD pilot is scored honestly** — the read-only 4×4 tree produced 16
  results, zero errors/holes, eight compensated cells, and eight degraded
  cells. The compensated vehicle scores are exactly **2.030809, 0.352059,
  0.061183, 0.227143** at updates 25/50/75/100. Update 25 therefore preserves
  vehicle transport relatively—about 7.08× the `final_1step` vehicle floor—but
  every vehicle remains below the absolute displaced gate. Eight of twelve
  non-vehicle cells are degraded, so their screen-space scores are diagnostic
  evidence of dissolving/churning output, not comparable compensated motion.
  See [[Displacement-Batch-Comparison]].
- **Coherence is now an independent calibrated axis** —
  `bench/coherence_metrics.py` scores every adjacent pair with DINOv2 feature
  similarity and real RAFT Small flow-warp residual, while native-resolution
  patch bulk/tails detect both structure loss and excess pointillist texture
  relative to frames 0–4. Raw flow magnitude is diagnostic only. Ten
  synthetics include fast-object, full-frame translation, and abrupt stable-
  texture-replacement discriminators;
  real DINOv2+RAFT full-frame translation is exactly **10/10**. The read-only
  26-clip calibration reproduces the expected group ordering: finals
  **9.796132** > four-step **5.673292** > one-step **0.969217**, with strict
  condition-tier separation also passing. SGMD update-25 vehicle is
  **7.553927** and decays **10.000000 → 8.763047 → 6.935005**; the pointillist
  ball is 0.000000. `--with-coherence` adds schema-4 two-axis matrices while
  the disabled schema-3 output tree remains byte-identical. See
  [[Coherence-Metric]].
- **Three SGMD training trajectories are now on that shared two-axis plane** —
  96 read-only clips across 24 checkpoints were scored into native sibling
  `batch_scores_2axis/` trees and one deterministic combined report/CSV. Motion
  means include camera-compensated cells only; degraded counts and denominators
  remain explicit. The proposed lower-lambda advantage does not reproduce:
  normalized lambda=0.2 beats lambda=0.05 on matched-prompt motion at four of
  five checkpoints and, at full-coverage step 50, reaches **1.693540 motion /
  7.403891 coherence** versus **0.735094 / 6.834893**. The large gains are
  prompt-specific and still require frame/blind fidelity review. The normalized
  lambda=0.1 step-25 vehicle is a valid decode with healthy coherence but a
  shortened/reversing traverse and a tracker/identity-confidence failure; its
  0.009206 score is not proof of zero visible motion. See
  [[SGMD-Training-Trajectories]].
- **P0 localization is accepted ground truth, not an experiment to repeat** —
  the recent-clean KV ring is dropped (it raised throughput while blind motion
  fell), resetting rolling-TAEHV state made motion worse, and the same latents
  tied under rolling TAEHV and full Wan decode. The decoder is exonerated; the
  remaining failure is latent-generator **motion-mode collapse**. Stage-2
  initialization versus Stage-3 DMD selection belongs to the orchestrating
  Model/Inference researcher, not this measurement track.

- **Frozen H100 runtime** —
  `bench/runtime/cf1-h100-cu128-v1.evidence.json` is SHA-256
  `8209043b4ebecc85f0e844f9c040b54fc1685104fe9e0b361ce9ee6d060b0c6c`.
  The schema-v2 frozen lock is SHA-256
  `d4d163d635ecbafb5b11bbe54cca7bdd5e9f80c1edc23ed96972821940ecc692`,
  has no unresolved fields, and binds CPython 3.12.3, CUDA 12.8, Torch
  2.8.0+cu128, NVIDIA driver 580.126.09, one H100 80GB HBM3, and actual
  FlashAttention-2. The locked runtime-environment identity is
  `858c16004d327fab276c8a8e43aa61bfdfcdbe78033e82edd96dfbe434839b7c`;
  native identity is
  `8884c4be14c4c7c404b346a76347ebd3ed8218c229b6066f4ceba743108c619c`;
  and the bound loaded-runtime probe identity is
  `070efce1bdcd5972fd9cdde7f02d66dab3320a568e7fb04caa2575aae8f0dd4c`.
  The lock binds the provider-observed OCI index, child manifest, and config
  digests rather than a mutable tag alone.
- **Frozen minimal CF++1 stack and real CUDA execution** — the exact source and
  11 registered assets remain byte-bound. The original frozen one-shot stack
  identity was
  `3fea725376946017add66c5ebc1684b9a8505c30ac15f2fd4af9b9e4e8cb0166`.
  Later `cf_cuda_session.py` guard hardening changed the guard-bundle bytes, not
  the frozen runtime/model inputs. Recomputing the identity formula reproduces
  the old value and yields the current stack identity
  `349c79f42726697310270a4e57395693cd16bab0ae7c903fd692dcfe995d5404`.
  This distinction matters: the worker correctly refused the stale expected
  stack instead of self-adopting current bytes.
- **Real exact CUDA smoke** — the complete smoke manifest at
  `bench/results/cf1-real-20260720-a/full-smoke/manifest.json` is SHA-256
  `d363fbe1b65b167a5cf731aa194b00676f23a4d6cdcdaa0f89cfe8a7c960ba99`.
  It records the exact 21-chunk topology `[1, 4 × 20]`, 81 decodable
  832×480 PNGs, exact 45-forward CF++1 generation, rolling TAEHV decode,
  synchronized device-to-host transfer, and clean completion. This is bounded
  development execution evidence, not a performance or quality result.
- **Runtime-tree contamination was detected and repaired** — the first real
  run created 1,649 bytecode files inside the frozen environment. They were
  removed only after an exact count/hash guard identified the generated set.
  The authorizer was replayed, and the subsequent full smoke produced zero
  changed environment files. Launches now disable bytecode before project
  imports; the frozen tree is checked again rather than silently accepting
  generated artifacts. A later clean replay-client relaunch failed closed on a
  second, smaller drift event: exactly 62 generated `.pyc` files totaling
  1,006,933 bytes under aiohttp and its dependencies. Package versions and
  source bytes had not drifted. Only after exact path/count/byte validation was
  that cache set removed; the complete static evidence check then returned to
  exact lock equality. The launch wrapper carries the no-bytecode guard. The
  remediated relaunch passed the unchanged frozen admission boundary and then
  passed exact final-client browser replay acceptance rather than adopting the
  changed tree.
- **Deterministic development video** — the 81 PNGs were assembled into an
  832×480, 16-fps, 81-frame, 5.0625-second H.264/yuv420p MP4 with no audio.
  Video SHA-256 is
  `8c892d63cbb4cec9ec7856a352630644c054e27159d092b7642291dc1a4f48fb`;
  artifact-manifest SHA-256 is
  `1b33d98978a413ef07817d82823bff81be91f6f44c56ac9e9f4d989d76359284`.
  Double encoding, exact probe checks, and full decode passed.
- **Development-only video understanding** — one explicit upload of those
  exact MP4 bytes completed through both Gemini 3.1 Pro Preview and TwelveLabs
  Pegasus 1.5. The combined manifest is SHA-256
  `e5f48cdfeb3412432f9a80844a04c8d2043498d96ccb8a0f8ad288ed7b3102a0`.
  Gemini identified unnatural motion, morphing, and sliding; Pegasus rated the
  same motion and anatomy much more favorably. That disagreement reinforces
  the existing evaluator-calibration blocker. These calls are explicitly
  development-only and authorize no quality or performance claim.
- **Persistent worker accepted on the frozen H100** —
  `bench/results/cf1-real-20260720-a/persistent-worker-acceptance.json` is
  SHA-256
  `30bfe91b4742bbd7e04966f9baed75562fce3c0a501b05c0063a45b7c54de115`.
  It binds current stack identity `349c79f4…` and worker-code identity
  `59127c702b5c7cd394e7f9921e22d09b4ce05d2e4ef14dbf2d4b56ab64228fc8`.
  Seeds 20260719 and 20260720 each completed 21 chunks/81 exact 832×480 PNGs
  on the same PID and worker-instance identity; the two seed outputs are
  distinct. The harness then sent an identity-fenced idle-worker `SIGKILL`,
  awaited failure detection, poisoned the backend and registry, reaped the
  worker, and proved a post-poison start is rejected synchronously.
- **Warm and output timings are scoped, not promoted** — model/runtime warm took
  517.196176814 seconds. The two bounded serial jobs reached all
  parent-validated PNGs in about 9.637 and 8.352 seconds, respectively. The
  manifest explicitly excludes MP4 encoding, network transport, provider
  upload, browser presentation, and paint; `performance_gate_evaluated` is
  false and `authorizes_performance_claim` is false. These values are useful
  diagnostics only.
- **Launch-environment false positive fixed** — the first worker attempt
  inspected sensitive-looking environment names after model imports; a benign
  library-added name could therefore look as if it crossed `exec`. The child
  now snapshots the exact launch environment before importing the model/runtime
  stack and reports that immutable pre-model result in `HELLO`. Red-first tests
  reproduce the old ordering failure and pin the new pre-model capture. The 80
  affected acceptance/worker/process tests pass, and an independent post-fix
  review is GO for this development-only lifecycle boundary.
- **Real CF++1 browser presentation accepted** — the genuine page explicitly
  identified the opt-in **CF++1 H100 backend / frozen runtime**, not the fake
  PNG sentence. One exact fox request used prompt SHA-256
  `fcaa04670647e417d16a4562d1027ca97a04ebcf9bed32c201fedd62ed9f785a`.
  Final DOM evidence was `streamState=complete`, `renderedFrames=81`,
  `renderedChunks=21`, `ackCount=21`, `expectedFrames=81`, and
  `serverCompleted=true`. Status text was exactly
  `Complete: 81 frames painted in 21 chunks; 21 presentation ACKs sent.` The
  canvas was present and visible:
  intrinsic 832×480, client 832×468, `display:inline`, `visibility:visible`,
  opacity 1, with aria-label `latest decoded frame`. The captured browser view
  visibly showed the generated fox frame and browser console errors were `[]`.
  This proves the real service-to-browser paint/ACK seam; it does not authorize
  quality, FPS, TTFF, or another performance claim. See
  [[Browser-Streaming-Transport]].
- **Manual motion diagnosis separated generation from playback** — the user's
  terse bouncing-ball test was not a frozen transport: five same-worker jobs
  each delivered the exact 21 chunks/81 unique PNGs with manifest-matching
  hashes and no cross-clip hash overlap. The terse seed-7 clip had adjacent
  8-bit MAD 2.229 and ball-center span only x=14 px/y=4 px; the ball stayed
  planted while its surface morphed. The explicit temporal-action prompt at
  seed 7 reached MAD 5.050 and x=110 px/y=152 px, and seed 20260719 reached MAD
  5.541 and x=64 px/y≥220 px. Both visibly bounced, but with drift,
  deformation, clipping, and nonphysical reversals. An upstream rich prompt
  reached MAD 6.891 with obvious articulated motion, while a generic
  action-emphasis suffix still did not make the terse ball bounce. The model is
  strongly prompt-conditioned and physically weak. Pegasus 1.5 independently
  scored the short clip 0/100 for motion and described it as static/no-bounce,
  while recognizing clear motion in the expanded clips. It also over-credited
  one expanded seed using physically impossible timestamps, so this is
  directional corroboration only and every judgment remains subordinate to the
  exact frames and measurements. See
  [[CF1-Prompt-Conditioned-Motion]].
- **The disclosed temporal-prompt compiler now fixes the terse product path** —
  with automatic resolution enabled, the raw browser input `a bouncing ball`
  resolved to a visible 106-word effective prompt that preserved one ball,
  fixed camera/framing, and the requested action while spelling out exactly
  three bounce arcs plus opening and closing holds. The raw and effective
  prompts remain distinct on the page, and exact/raw mode remains available;
  a one-time resolution identifier fences the subsequent start. On the live
  CF++1 H100 path, that resolved request completed 81 painted frames in 21
  chunks with 21 presentation ACKs. The 16-fps local replay visibly showed the
  ball leave the floor and return, while the browser console had no warnings or
  errors. This is the intended Serving/Product workaround under
  [[ADR-003-serving-product-model-track-split]], not a model/inference change or
  a general motion-quality claim.
- **The JPEG-q90 serving regression is fixed at the parent/worker seam** — the
  production worker emitted valid boot-bound `image/jpeg` payloads, but
  `ProcessStreamingBackend` still hard-coded `image/png` while validating the
  first returned chunk. It therefore rejected the valid stream and poisoned
  the backend before presentation. The parent now owns a validated media-type
  hook for its default PNG contract; `CF1ProcessStreamingBackend` overrides it
  narrowly for the boot-bound `jpeg-q90-cpu-v1` profile and validates every
  JPEG payload. The regression was reproduced red-first and the exact test now
  passes both locally and in the frozen remote runtime. See
  [[Browser-Streaming-Transport]].
- **Prompt enhancement is now explicit, optional, and editable** — the page has
  one prompt textarea as its only source of truth. `Generate` sends exactly the
  text currently in that field. `Enhance prompt` separately invokes
  `gemini-3.1-flash-lite` through transform
  `gemini-3.1-flash-lite-video-prompt-expansion-v2` and replaces the same field,
  so the user can inspect, edit, or discard the result before generation.
  Editing invalidates unused enhancement provenance, but cannot invalidate the
  one-time resolution fence already attached to a start that has been sent.
- **The single-field prompt lifecycle is race-fenced through completion** — a
  late resolver response cannot overwrite text edited by the user or the prompt
  of a generation already started. A generation remains busy until terminal
  presentation, preventing another start during an active job, and a specific
  terminal protocol error survives the socket-close event instead of becoming
  a generic disconnect. The executable browser client suite passes **29/29**.
- **The optional-enhancement UI passed the live browser surface** — clicking
  `Enhance prompt` for `a bouncing ball` replaced the same editable field in
  **1730 ms** without starting generation. The reviewed enhanced text then
  completed the genuine path at 81 painted frames / 21 chunks / 21 presentation
  ACKs, and bounded replay remained visible on the loopback page. See
  [[Browser-Streaming-Transport]].
- **The final actual-file CLI panel returned 3/3 SHIP** — Claude Opus, Kimi,
  and Grok found no blocking defect in the single-field UI or its race fences.
  Claude's conditional concern about the `/demo.js` asset alias was resolved
  by the same live browser acceptance, which loaded and exercised that route.
  Remaining suggestions were browser-floor/accessibility polish, not blockers
  for this user-visible release.
- **Completed clips now replay instead of freezing on frame 81** — the old
  page decoded, painted, and closed each bitmap once, discarded history, and
  left the terminal frame on the canvas. The client now retains encoded frames
  under a 64 MiB replay budget and loops them locally at 16 fps after successful
  presentation. Replay has independent pacing and epochs, never changes the
  original frame/chunk/ACK counters, and degrades independently if replay-only
  decode fails. Job replacement, pre-completion disconnect/error/cancel, and
  stale in-flight paints are fenced; a post-completion socket close preserves
  the local loop. Before any paint, the client requires exactly 21 latent
  chunks, 81 RGB frames, topology `1 + 20×4`, exact chunk/frame offsets, and a
  complete 21/81 terminal sequence. Browser input is capped at 16 MiB per
  chunk, the uint32 seed is user-visible and validated, and the page discloses
  that terse prompts may produce little motion. See
  [[Browser-Streaming-Transport]].
- **The synchronized replay build is accepted in a real browser** — one exact
  detailed bouncing-ball request at visible seed 7 reached
  `streamState=complete`, `serverCompleted=true`, 81 rendered frames, 21
  rendered chunks, and 21 original presentation ACKs. It then entered
  `replayState=replaying`. Full-page browser screenshots 500 ms apart had
  different SHA-256 values while the 81/21/21 counters remained unchanged;
  the two views visibly placed the ball at different vertical phases. Browser
  console errors were `[]`. This proves bounded local replay of the exact
  synchronized build, not motion quality, FPS, TTFF, or sustained service.
- **Current validation** — focused browser/worker/acceptance/security tests pass
  **72/72**; the affected worker/process/acceptance set passes **80/80**. Full
  discovery on Python 3.12.9 is **434/435**. The sole failure is unrelated and
  preexisting: `test_lone_surrogates_and_recursive_json_are_stable_command_errors`
  expects `invalid_json` for 1,100 nested JSON arrays, while this Python parses
  them and the command boundary returns `invalid_command`.
  The newer replay/topology client suite passes **19/19** and the targeted
  Python WebSocket/server suites pass **27/27** (22 WebSocket plus 5 launch
  lifecycle tests). Red-first coverage includes replay,
  visible seed propagation, pre-completion stale-paint fencing, fixed topology,
  premature completion, malformed chunks, and the browser byte cap.
  The released temporal compiler's focused suite passes **14/14**. The new
  executable browser client suite passes **21/21**. The new parent media-type
  hook and CF1 JPEG override pass in the **39-test** local process/worker set,
  and the exact red-first JPEG regression also passes in the remote Python
  3.12 runtime used by the live H100 worker.
- **Historical evidence remains development context** — the archived 31.039
  fps rolling-TAEHV result and legacy 5.67/10 CF1 quality result used different
  decoder/artifact paths. They do not become one exact-stack result merely
  because real generation and lifecycle acceptance now pass.
- **The studio now has a working self-hosted "Open model" mode (separate
  openstudio track)** — `studio/src/lib/open-realtime.ts` implements the
  openstudio wire contract v1 client (13/17-byte little-endian JPEG framing,
  hello/proto gate, time-based 15-fps ticker with five skip-gates and
  newest-frame-wins, 2-frame warmup discard, prompt coalescing acks, 10 s ping,
  5 s stall indicator, pagehide teardown), and `studio/src/routes/studio.tsx`
  gained an additive mode toggle + ws-URL field (localStorage-persisted) while
  the Lucy branch of `start()` is diff-verified unchanged. Verified against
  `openstudio-server --pipeline fake` in a real browser: live at 15.0 fps
  (HUD `15.0 fps · gpu ~64 ms · e2e ~68 ms`), prompt hot-swap visibly changes
  output and hits the server log, second client gets busy, stop/tab-close free
  the single slot, server freeze shows `stalled` then recovers, and an
  intercepted recording ffprobes as vp9 512×512 webm. `pnpm typecheck` green;
  header codecs pinned to Python struct vectors in a 7/7 vitest. No GPU spend.

- The standing review surface is the standalone CLI only, never an MCP tool.
  Invoke the exact three-lineage panel with:
  `consensus run --lineages claude-opus,kimi,grok --count 3 --json < prompt`.
  The `claude-opus` lineage is configured to try Claude Fable first and fall
  back automatically to Claude Opus. Do not request `claude-fable` directly,
  substitute `kimi-k3`, or build a manual fallback chain.
- The session-21 actual-file CLI panel completed with concrete Claude Opus 4.8,
  Kimi 2.6, and Grok 4.5. All three accepted the core metric as a sound
  development scorer. Their identical-`m07`/`k14` concern is resolved by the
  read-only inventory: those clips are intentionally byte-identical, so the
  inferential correlation is now deduplicated to ten contents. Their valid
  knife-edge boolean concern is fixed by the calibrated conservative 5.0
  threshold. Fixture arithmetic remains a calibration snapshot; the red-first
  synthetics and live learned-backend runs are the algorithm/runtime evidence.
- The session-22 actual-file CLI panel was partial. Kimi 2.6 returned a review;
  Claude Opus 4.8 timed out but left useful reasoning recovered from
  `ses_07b9e75c3ffeKA2ElSp4Ouq1BQ`; Grok 4.5 failed for insufficient provider
  balance. Kimi and recovered Opus accepted the core batch contract and found
  positional/Markdown and uppercase nested-file gaps that are fixed. A local
  actual-file reviewer then found stale output and output-root contamination;
  both reproduced red and are fixed. This is independent review, not completed
  three-lineage consensus.
- The session-23 actual-file CLI panel was also partial: Claude Opus 4.8
  returned a high-confidence PASS and independently verified the 7×4 inventory,
  vehicle spot values, and all comparison arithmetic; Kimi and Grok failed for
  provider balance. A separate local adversarial reviewer found that the new
  per-clip continuation initially downgraded `OSError` too broadly; its
  propagation and unchanged-output behavior reproduced red and is fixed. This
  is one external advisory plus local review, not three-lineage consensus.
- The session-24 actual-file CLI panel was likewise partial: Claude Opus 4.8
  returned **NO FINDINGS** after tracing the narrow camera-fallback gate,
  fallback-only report markers, error/abort semantics, default degraded
  exclusion, stale-output replacement point, and exact SGMD payload. Kimi and
  Grok failed for provider balance. A separate local actual-file reviewer also
  found no issue and confirmed all eight pre-existing SGMD spot reports are
  byte-identical to the new per-clip batch JSON. This is one external advisory
  plus local review, not three-lineage consensus.
- The session-19 actual-source-and-tests CLI panel returned **3/3 SHIP** from
  Claude Opus 4.8, Kimi, and Grok 4.5. All three independently accepted the
  parent media-type hook, narrow boot-profile JPEG override, prompt provenance
  path, and red-first regression as shippable. The one confirmed product
  follow-up is mid-generation disconnect/cancel recovery: the WebSocket session
  cancels the active registry job, the process backend intentionally becomes
  `stopped`, and the demo currently warms only at server boot. A refresh during
  generation therefore requires a server restart before the next job; do not
  confuse this bounded follow-up with a blocker for the completed live run.
- The old MCP/OpenCode bridge, its outer timeout behavior, and SQLite
  session-recovery procedure are obsolete operating instructions. The older
  session IDs below are retained only as historical review evidence; they do
  not describe how a new consensus review should be run.
- The latest diff-fed review parent was
  `ses_07eba6ca6ffeZao49EpwVikquA`. The outer call timed out. Fable child
  `ses_07eb96845ffejMINZII6AmS8XV` produced zero output/tokens and no rate-limit
  signal, so automatic Opus fallback correctly did not run. Kimi child
  `ses_07eb96842ffeDnxSt5CNEmht9B` and Grok child
  `ses_07eb96841ffelPmwu896DGqDib` both returned GO for one bounded persistent-
  worker acceptance followed, only if it passed, by one localhost browser
  acceptance. Their recorded costs were $0.310242 and $0.107124, totaling
  **$0.417366**. This is two independent GO positions recovered from a timed-out
  panel, not a synthesized three-voter verdict.
- The environment-snapshot correction and other reproducible review findings
  were fixed red-first. Review and the browser observation never override the
  worker manifest's explicit quality/performance claim fences.
- The latest actual-file replay/topology review parent is
  `ses_07e3913c1ffeJA1r0XbUltbgDs`. The concrete Fable child
  `ses_07e38d318ffeKb3Tm59uxZn0TR` timed out with no output and no rate-limit
  signal, so automatic Opus fallback correctly did not run. Kimi child
  `ses_07e38d315ffeMGw3O0GnCGfvyP` timed out without a final. Grok 4.5 child
  `ses_07e38d314ffenrE2Ihkvz4kqyJ` returned a BLOCK advisory for stale paints
  after pre-completion disconnect/error, absent expected-topology validation,
  missing negative tests, and no browser payload cap. Those findings were
  reproduced red-first and fixed, along with stricter exact `21/81` and
  `1 + 20×4` enforcement. A separate local actual-file reviewer found no
  remaining ship blocker and all 19 Node tests pass. This is one completed
  external advisory plus one local review, **not** a three-lineage consensus.

## Operational state

- The prior session-20 browser release remains banked, but dedicated pod
  `4ddf7fhj9bdbam` is **EXITED** and cannot restart because its host has no free
  GPU. The loopback server is gone and the frozen volume is unreachable. Do not
  reconnect, retry a start, reprovision the pod, or rebuild that frozen stack on
  another host. This session used only local MPS/CPU measurement and did not
  touch any pod.
- No new RunPod amount is entered because an exact billed total for this
  sequence is not available. Do not infer spend from elapsed estimates.
- The only new known review spend entered in the ledger is **$0.417366** for
  the two completed consensus voters above.
- Local provider credentials remain secret and untracked. No credential value,
  resolved authorization header, or `.env` content may enter consensus prompts,
  manifests, or the brain.
- No git commit was created.

## Honest gaps

- The accepted browser run proves one short localhost presentation, not a
  sustained browser-performance result, remote multi-client service, or public
  deployment. Its DOM counters and visible canvas are presentation evidence;
  they do not make the worker's serial PNG-readiness timings into FPS or TTFF.
- The accepted worker manifest is lifecycle evidence, not a benchmark. Its
  serial PNG-readiness timings cannot be reported as FPS, TTFF, browser latency,
  or a MotionStream comparison.
- Gemini remains transport-verified/calibration-pending and Pegasus remains
  calibration-failed. The replacement protocol still needs qualified judge
  families, the registered human audit, complete Round-A/Round-B evidence, and
  a completed project-level GO review.
- A sustained ≥60-second exact-stack run remains unproved. The registered
  241-latent sentinel is still incompatible with the pinned 21-latent
  global-attention cache and needs a genuinely long-horizon-capable policy or
  an honest protocol revision.
- The public quality-qualified ≥29-fps H100 headline remains blocked. The real
  smoke, development upload, and lifecycle acceptance authorize none of its
  quality or performance components.
- The displacement calibration is small: 11 labels but only 10 independent
  contents because `m07` and `k14` are the same bytes. Its deduplicated
  correlation is positive but not conventionally significant at p=0.062, and
  the metric deliberately under-represents in-place articulated gait. Retain
  blind review for semantics and use the harness as a repeatable first screen.
- A positional batch with heterogeneous filenames can infer only trailing
  holes; missing-middle identity requires a complete shared filename template.
  `slot_mode` records that boundary instead of inventing a mapping.
- A displacement score is not a prompt-action score. The fork-grid
  `final_2step` ball's one-time lateral recenter and barrel's slow drift/pan
  produce high coherent-displacement values without the requested bounce/roll
  dynamics. Seven other valid-grid cells are explicitly unscorable. Do not
  zero-fill errors, compare unequal coverage as if complete, or cite the full
  matrix as proof that every pre-DMD condition beats both finals.
- A starred SGMD value is screen-space-only because the clip could not support
  camera compensation. It remains useful for diagnosing churn, but it must not
  enter a compensated mean by default or be cited as prompt-faithful transport.
  Update-25 vehicle is a large relative improvement, not an absolute pass:
  score 2.030809 still has `displaced=false`. Eight of twelve non-vehicle cells
  are degraded, consistent with dissolution rather than a usable checkpoint.
- The coherence calibration is 26 clips from two related experiment families,
  not a broad human-rated corpus. Its early-relative spatial guards also assume
  the first five frames are a usable reference, and 18/26 rows sit at a hard
  0-or-10 floor. Use it to separate clean/mixed/collapsed operating regions,
  not to finely rank already-collapsed texture soups or clips bad from frame 0.
- `degrades_over_time=false` does not mean healthy: a clip already incoherent
  in frames 0–20 cannot show a later two-point drop. Read all three segment
  scores and retain frame/blind review for prompt fidelity and physics.
- Equal compensated-cell counts do not imply equal prompt membership. The
  lambda sweep has 15 compensated cells per arm, but pooled means use different
  subsets; the matched-prompt table is the apples-to-apples comparison. High
  lambda's very large vehicle/barrel and step-30 ball values still require
  visual/blind validation before promotion.
- The normalized lambda=0.1 vehicle anchors at steps 15 and 20 reproduce
  numerically but are screen-space fallbacks, not compensated measurements.
  Their values are excluded from compensated trajectory means.

## Likely next experiment

Stop at the completed measurement handoff. Lambda=0.2 is the strongest
quantitative promotion candidate in this fixed sample, but the orchestrating
researcher should first inspect/blind-rate its high-motion cells for prompt
fidelity, direction, and physics and, if the isolated normalized step-25 dip
matters, add another deterministic seed plus a manual or box-assisted track
check. This measurement track should not choose training or inference strategy
or re-enter KV-ring, decoder, AvatarForcing, or inference-schedule work.

Do not touch the exited H100 pod. The next bounded measurement increment should
begin only when the orchestrator supplies a concrete experiment output or a new
measurement question.
