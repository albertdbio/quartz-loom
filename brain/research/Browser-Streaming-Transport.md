---
type: research
status: active
date: 2026-07-20
description: "The live browser path now exposes one user-owned editable prompt: Generate sends its exact current text, while optional Gemini 3.1 Flash-Lite enhancement replaces that same field for review/edit. Prompt provenance, generation-busy, late-response, and terminal-error fences pass 29/29 client tests. Live `a bouncing ball` enhancement completed in 1730 ms, followed by 81 painted frames / 21 chunks / 21 presentation ACKs and visible loopback replay. This remains bounded development acceptance, not quality or performance authorization."
anchors: ["bench/streaming_process.py#ProcessStreamingBackend", "bench/cf_streaming_worker.py#CF1ProcessStreamingBackend", "bench/streaming_websocket.py#BrowserStreamingServer", "bench/streaming_websocket.py#_WebSocketSession", "bench/cf_streaming_web_demo.py#_run", "bench/static/streaming_demo.js", "bench/tests/streaming_demo_client.test.mjs", "bench/tests/test_cf_streaming_worker.py"]
related: ["[[State]]", "[[Streaming-Service-Boundary]]", "[[Pinned-CF1-CUDA-Bootstrap]]", "[[CF1-H100-Runtime-Preflight]]", "[[CF1-Persistent-CUDA-Worker]]", "[[CF1-Prompt-Conditioned-Motion]]", "[[Gotcha-Transport-Write-Is-Not-Presentation]]", "[[Gotcha-Async-Timeouts-Need-Task-Isolation]]", "[[ADR-003-serving-product-model-track-split]]", "[[session-7-browser-streaming-transport]]", "[[session-17-frozen-h100-generation-and-browser-acceptance]]", "[[session-18-motion-diagnosis-and-looping-replay]]", "[[session-19-live-temporal-prompt-jpeg-release]]", "[[session-20-editable-optional-prompt-enhancement]]"]
---

# Browser streaming transport

## Bottom line

The local fake/dev stream and the separate opt-in real CF++1 backend reach a
real browser over binary WebSockets. The genuine page identified the **CF++1
H100 backend / frozen runtime** and completed the exact 81-frame presentation
contract. The initial client was nevertheless a one-pass canvas: after
completion it left only frame 81 visible, which made a completed clip look
static to a user arriving after the five-second presentation.

The client now presents incoming frames at 24 fps, retains a bounded copy of the
encoded clip, and starts a local 16-fps loop only after successful original
presentation. Exact capture separately proved that the raw terse bouncing-ball
prompt has almost no ball motion; the disclosed temporal compiler now repairs
that product path without changing the model. The accepted live JPEG-q90 run
completed 81/21/21 and visibly replayed the ball leaving the floor and
returning. Neither presentation nor replay is evidence for CUDA throughput,
quality, or performance-authorized first-visible latency.

The current product surface no longer applies enhancement automatically or
shows a second read-only effective-prompt field. One editable textarea is the
source of truth: `Generate` sends its exact contents, and the optional
`Enhance prompt` action replaces those contents for user review and further
editing before any start. This keeps prompt authorship visible without giving
up the motion-oriented helper.

## Wire contract

- The server binds only to a literal loopback IP and requires exact Host,
  Origin, and `realtime-video.websocket.v1` subprotocol values.
- A GET of `/` issues one bounded, expiring, single-use HttpOnly SameSite nonce
  cookie. The nonce never appears in the URL, page source, or logs.
- Start is fixed to `{type, prompt, seed}` and the short 21-latent/81-RGB job.
- The production browser profile is boot-bound as `jpeg-q90-cpu-v1`. Its worker
  advertises `image/jpeg`, and the parent validates that media type and every
  JPEG payload before constructing a decoded chunk. The default process
  contract remains strict PNG; the CF1 override is narrow to its immutable boot
  profile.
- A `chunk` text header is followed by exactly `frame_count` binary raster
  messages. The header intentionally contains no delivery ID.
- After the complete binary group is written and registered, the server emits
  `chunk_committed` with an unpredictable delivery ID. Only that token is
  acknowledgement-eligible.
- At most two chunks may await presentation. Missing ACKs time out into a
  sanitized `client_backpressure_timeout`, retire connection state, and allow a
  fresh start when the socket remains usable.
- Job IDs and delivery IDs cannot be reused over the bounded lifetime of a
  connection. Replacement and cancellation are fenced by job ID.

## Client presentation semantics

The browser decodes each encoded raster with `createImageBitmap`, paces the
initial live pass at 24 fps, draws one frame per animation-frame opportunity,
then waits for a following animation frame before incrementing painted counters
and eventually sending the chunk ACK. The server can enforce only
registration/commit order; the same-origin client is the component asserting
paint/presentation order.

Completion is announced only when the server terminal event has arrived and
all 81 frames have been painted. DOM counters expose rendered frames, chunks,
ACK count, expected frames, terminal state, and replay state for browser
verification.

The current client fails closed before paint unless start acceptance declares
exactly 21 latent frames and 81 RGB frames, and every chunk matches the fixed
`1 + 20×4` count/index/first-frame-offset topology. A completion event before
all 21 chunks/81 frames is a protocol error. Incoming browser payload is capped
at 16 MiB per chunk.

After original completion, replay uses a separate pacing object and a separate
job/epoch fence. It never increments the original rendered-frame,
rendered-chunk, or ACK counters and never emits another `presented` command.
Encoded frames are retained only up to a 64 MiB total budget. Budget overflow
or replay-only decode failure disables replay without destroying already-proven
completion. Replacement, pre-completion error/disconnect/cancel, and stale
in-flight paints clear the retained clip. A socket close after completed
presentation leaves local replay intact.

## Real CF++1 acceptance

One in-app-browser run used the exact fox prompt with SHA-256
`fcaa04670647e417d16a4562d1027ca97a04ebcf9bed32c201fedd62ed9f785a`.
The page text identified the opt-in CF++1 H100 backend and frozen runtime; it
did not contain the fake-backend description.

Final DOM evidence was:

- `streamState=complete`;
- `renderedFrames=81` and `expectedFrames=81`;
- `renderedChunks=21`;
- `ackCount=21`; and
- `serverCompleted=true`.

Status text was exactly
`Complete: 81 frames painted in 21 chunks; 21 presentation ACKs sent.` The
canvas was present with intrinsic dimensions
832×480 and client dimensions 832×468. Computed presentation state was
`display:inline`, `visibility:visible`, and opacity 1; its aria-label was
`latest decoded frame`. The captured browser view visibly showed the generated
fox frame. Browser console errors were `[]`.

The first acceptance instance then shut down gracefully, removing its remote
server and worker processes. The user later exercised a fresh manual-test
surface and reported that “a bouncing ball” looked static. Exact captures
separated the real prompt-conditioned motion weakness from the final-frame
playback defect. The final replay HTML/CSS/JavaScript bytes have been
synchronized. The initial clean relaunch failed closed on 62 generated `.pyc`
files / 1,006,933 bytes in the frozen environment, with no package/source
version drift. After exact path/count/byte validation, only that cache set was
removed, full static evidence returned to the exact lock, and the no-bytecode
wrapper relaunched cleanly. That exact build then passed real-browser replay
reacceptance; see [[CF1-H100-Runtime-Preflight]].

### Temporal-prompt JPEG-q90 live release

The raw browser input `a bouncing ball` was resolved by the disclosed temporal
compiler into a visible 106-word effective prompt specifying exactly three
bounce arcs. The first attempt reached the real worker but failed before
presentation: `ProcessStreamingBackend._run_claimed_job` still hard-coded
`image/png`, so it rejected the worker's valid, boot-bound `image/jpeg` first
chunk and poisoned the backend.

The fix introduced `ProcessStreamingBackend._validated_frame_media_type` for
the parent default PNG contract and a `CF1ProcessStreamingBackend` override
that accepts only `image/jpeg` under `jpeg-q90-cpu-v1` and validates every
payload as JPEG. A red-first regression reproduced the exact mismatch before
the implementation and now passes locally and in the remote Python 3.12 worker
runtime.

After the fix, the same raw browser request completed the genuine CF++1 H100
path end-to-end: 81 frames painted, 21 chunks committed, and 21 presentation
ACKs sent. The browser then replayed the retained clip; direct observation
showed the ball leave the floor and return. Console warnings and errors were
both empty. This is the Serving/Product release gate in
[[ADR-003-serving-product-model-track-split]], not a model/inference experiment.

### Single-field optional enhancement release

The prior automatic raw/effective dual-field UI is superseded by one editable
prompt textarea. `Generate` always sends exactly the field's current text. The
separate `Enhance prompt` action invokes `gemini-3.1-flash-lite` with transform
`gemini-3.1-flash-lite-video-prompt-expansion-v2`, then replaces the same field
without starting a job. The user therefore reviews, edits, or discards the
effective prompt as ordinary text.

Enhancement provenance remains single-use but follows the user's action rather
than owning the field. Editing invalidates any unused provenance. Once a start
has been sent, later edits do not invalidate the one-time resolution fence
already bound to that request. Pending resolver responses are retired when the
field changes or generation begins, so a late response cannot overwrite a user
edit or an already-started prompt.

The client also treats generation as busy through terminal presentation rather
than only through start acceptance, closing the second-start window. Specific
terminal protocol errors remain terminal when the socket subsequently closes;
the close handler cannot replace them with generic disconnected state.

Live browser acceptance entered `a bouncing ball`, invoked enhancement, and
observed the replacement in the same editable field after **1730 ms** without a
generation start. Generating the reviewed text then completed at 81 painted
frames / 21 chunks / 21 presentation ACKs, after which replay remained visible
on the loopback page.

## Motion report and replay correction

The short-prompt capture delivered 81/81 unique, manifest-matching frames, yet
the ball center moved only 4 px vertically and never left the floor. A detailed
temporal prompt at the same seed moved the center 152 px vertically and visibly
bounced; another seed moved it at least 220 px. A generic action-emphasis suffix
still did not make the short prompt bounce. Therefore:

- WebSocket and browser transport were not freezing or duplicating frames;
- the raw one-step generator is highly sensitive to prompt specificity;
- the old page still made diagnosis worse by discarding the clip after its
  single pass; and
- local replay is necessary for inspection but cannot repair weak action
  obedience or physics.

The page now exposes and validates the uint32 seed and uses one editable prompt
as the source of truth. `Generate` sends its exact current text. Optional
`Enhance prompt` discloses temporal expansion, replaces that same field for
review or editing, and never starts a job. Either the user's original text or
the reviewed replacement then passes unchanged through the worker path; there
is no hidden rewrite after the explicit action.

## Review-driven race fixes

The first actual-file consensus attempt timed out, but Grok's completed child
session was recovered from parent `ses_0818704d9ffeNs36KPw31kgHEy`. It found
three real replacement races:

1. a stale job retirement interpreted a fenced `False` send as socket failure
   and disconnected the replacement;
2. a binary group retired during replacement still received an unconditional
   commit token;
3. the browser treated an already-stale commit as a fatal current-protocol
   error.

All three reproduced red before their localized fixes. A fourth red regression
made delivery-ID uniqueness cover the full bounded connection rather than only
the 64 most recently retired tokens. The stale-send finding was then searched
across siblings and reproduced/fixed in the NDJSON adapter too.

The old browser test only searched JavaScript source strings. The executable
Node harness now runs the actual client with mocked DOM/WebSocket/rAF surfaces,
proving that stale commits are ignored, current commits without metadata remain
fatal, and `presented` is not sent until after draw plus a later presentation
opportunity.

A smaller post-fix review also timed out at the outer bridge, but SQLite
recovery from `ses_08178b6e8ffezbHmo49X5yKZKd` retained concrete
`claude-fable-5` and `grok-4.5` finals; Kimi-K3 had no final. Both completed
voters accepted all five fixes for this fake milestone. Fable requested one
missing replacement-before-predecessor-commit test, which now executes the real
client and passes. Grok's remaining exception-cleanup concern came from the
excerpt ending before the existing cleanup, retirement, notification, and
re-raise lines, so it was rejected against disk.

The replay/topology actual-file review parent
`ses_07e3913c1ffeJA1r0XbUltbgDs` also timed out. Concrete Fable child
`ses_07e38d318ffeKb3Tm59uxZn0TR` produced no output and no rate-limit signal,
so automatic Opus fallback correctly did not run. Kimi child
`ses_07e38d315ffeMGw3O0GnCGfvyP` produced no final. Grok 4.5 child
`ses_07e38d314ffenrE2Ihkvz4kqyJ` returned a useful BLOCK advisory: fence stale
paints on pre-completion disconnect/error, validate expected topology to avoid
an empty/tight replay path, add missing negative tests, and cap browser input.
Each reproducible item was fixed red-first. The final client goes further by
enforcing exact `21/81` and `1 + 20×4` topology. A separate local actual-file
review found no remaining ship blocker. Because only one external voter
completed, this is not a multi-lineage consensus verdict.

## Verification

- Focused WebSocket tests: 20/20.
- Executable Node client tests: 4/4, including replacement before predecessor
  commit.
- Current replay/topology executable Node client tests: **19/19**. They cover
  visible seed propagation and rejection, prompt-risk disclosure, stale commit,
  disconnect/cancel paint fencing, exact accepted and chunk topology, premature
  completion, replacement, payload and replay limits, completion-counter
  preservation, looping, and replay-only decode failure.
- Current targeted Python WebSocket/server tests: **27/27** (22 WebSocket + 5
  launch lifecycle tests).
- Persistent-process tests: 35/35; the process seam is verified separately.
- Combined process/service/NDJSON/WebSocket tests: 113/113.
- Current focused browser/worker/acceptance/security suite: **72/72**.
- Current affected worker/process/acceptance suite: **80/80**.
- Full discovery on Python 3.12.9: **434/435**. The sole failure is the
  unrelated preexisting recursive-JSON expectation: the test expects
  `invalid_json` for 1,100 nested arrays, while this Python parses them and the
  command boundary returns `invalid_command`.
- Real in-app browser: 81 painted frames, 21 chunks, and 21 ACKs in 5.092
  seconds, with zero console warnings/errors.
- Same-socket in-flight replacement: replaced after five painted frames; the
  replacement completed 81/21/21 in 5.154 seconds with zero warnings/errors.
- Exact synchronized replay build: one detailed bouncing-ball request at seed 7
  completed 81/21/21 with `serverCompleted=true`, then remained
  `replayState=replaying`. Browser screenshots 500 ms apart had different
  SHA-256 values and visibly different ball phases while all original counters
  stayed fixed; console errors were `[]`.
- Temporal compiler: focused prompt-resolution suite **14/14**.
- Current executable browser client suite: **21/21**.
- JPEG media-type regression: parent/process/worker suite **39/39** locally;
  exact `test_real_backend_accepts_only_its_boot_bound_jpeg_payloads` passes in
  the remote Python 3.12 runtime.
- Live disclosed raw-input run: `a bouncing ball` resolved to a 106-word
  effective prompt; the genuine CF++1 H100 JPEG-q90 path completed 81 painted
  frames / 21 chunks / 21 ACKs, replay visibly showed departure from and return
  to the floor, and browser console warnings/errors were both empty.
- Current single-field executable browser client suite: **29/29**, including
  exact-text direct generation, explicit enhancement, edit/provenance fencing,
  late-response retirement, active-generation exclusion, and terminal error
  preservation across socket close.
- Current optional-enhancement live acceptance: `a bouncing ball` was enhanced
  into the same editable field in **1730 ms** without starting generation; the
  reviewed prompt then completed 81 painted frames / 21 chunks / 21 ACKs and
  left replay visible on the loopback page.
- Final attached-file CLI consensus: Claude Opus, Kimi, and Grok returned
  **SHIP**. Claude's conditional `/demo.js` alias question was resolved by the
  live accepted page loading and exercising that route.

## Remaining boundary

The original bounded real service-to-browser path, synchronized replay, and the
disclosed temporal-prompt/JPEG-q90 product path are accepted for one localhost
job each. The latest replay changed the visible ball position after completion
without changing the original 81/21/21 counters or sending duplicate
presentation ACKs.

Even after that, this is not a benchmark or deployment proof. It covers one
localhost job, not sustained service, multi-client operation, public hosting,
or remote-network behavior. DOM paint, ACK, and replay evidence do not attach
the archived 31.04-fps result to this runtime, authorize TTFF/P95, or qualify
the generated video's motion quality.
