---
type: research
status: active
date: 2026-07-20
description: "The separate warm-only CF++1 worker has passed frozen-H100 lifecycle and browser acceptance: two distinct-seed jobs reused one PID/instance, forced death proved poison/reap, and a fresh genuine page painted 81 frames in 21 chunks with 21 post-paint ACKs. Quality and performance remain unauthorized."
anchors: ["bench/cf_streaming_worker.py#CF1StreamingWorker", "bench/cf_streaming_worker.py#CF1ProcessStreamingBackend", "bench/cf_streaming_process_worker.py#run", "bench/streaming_process.py#ProcessStreamingBackend", "bench/cf_worker_acceptance.py#run_cf1_worker_acceptance"]
related: ["[[Research]]", "[[State]]", "[[CF1-H100-Runtime-Preflight]]", "[[CF1-Latent-Pull-and-Smoke]]", "[[CF1-Development-Video-Artifact]]", "[[Streaming-Service-Boundary]]", "[[Browser-Streaming-Transport]]", "[[session-14-persistent-cuda-worker]]", "[[session-15-persistent-worker-acceptance]]", "[[session-16-acceptance-evidence-hardening]]", "[[session-17-frozen-h100-generation-and-browser-acceptance]]"]
---

# CF++1 persistent CUDA worker

## Bottom line

The real path now has a distinct, non-default process worker instead of a fake
worker replacement. `CF1ProcessStreamingBackend` must be explicitly warmed;
its child builds and verifies the exact CF++1 runtime before `HELLO`, owns one
model/GPU instance, and maps each accepted `NEXT` credit to exactly one latent
pull, one rolling decode, and one PNG `CHUNK`. A clean terminal credit at index
21 finalizes the decoder and generator before `COMPLETE`, then permits a fresh
per-job session on the same warm runtime.

The runtime/evidence lock is now frozen and the real child passed `HELLO` with
stack identity `349c79f4…` and worker identity `59127c70…`. Seeds 20260719 and
20260720 each completed the exact 21-chunk/81-PNG topology on the same PID and
worker-instance identity with distinct outputs. The acceptance harness then
sent an identity-fenced idle-worker `SIGKILL`, awaited the death probe, poisoned
the backend and registry, reaped the worker, and proved a fourth start is
rejected synchronously.

`scripts/cf-streaming-acceptance` now makes the first real-H100 lifecycle proof
one reviewed command instead of an operator recipe. It uses the actual bounded
`StreamingJobRegistry`/emitter path, reconciles independently received payload
hashes with child completion evidence for two clean jobs on one PID/instance,
then invokes an identity-fenced idle-worker process-group death. A third awaited
job must detect that death, emit `backend_fatal`, poison/reap the backend and
poison the registry; a fourth start must be rejected synchronously. Only then,
and only after backend close, is one no-clobber fsynced JSON record published.
That published record is SHA-256
`30bfe91b4742bbd7e04966f9baed75562fce3c0a501b05c0063a45b7c54de115`
with `status: accepted` and every performance/quality/browser/upload/additional-
GPU authorization flag false.

## Admission and provenance

The parent execs the real entrypoint with the interpreter's resolved path and
`-I`. The child installs only the resolved project root. A local
`bench/__init__.py` makes that first path a regular package, so a later regular
site package cannot shadow the project namespace. Before project imports, a
directly executed child redirects bytecode lookup to a random, absolute,
guaranteed-nonexistent prefix and disables bytecode writes; a timestamp-valid
local `__pycache__` file therefore cannot substitute bytes for the source that
was hashed.

The launch environment is an explicit allowlist containing only
`PYTHONUNBUFFERED=1` and the canonical OCI image digest. Sensitive-looking
names are rejected at construction and revalidated from a copied snapshot at
the final pre-`Popen` boundary. The specialized backend locks its stack,
entrypoint, arguments, bundle, environment, warm requirement, latent cap,
every timeout, every wire/payload/prompt/job bound, and stderr retention
against post-construction downgrade.

The factory requires an externally frozen `expected_worker_code_sha256`; it
does not authorize whatever bytes happen to be present. The parent rejects a
different current bundle before `Popen`, passes the expectation explicitly,
and a standard-library-only child prelude independently recomputes it before
any project import. `worker_code_sha256` now binds the complete 15-file local
execution closure: package initializer, real entrypoint, engine wrapper,
parent supervisor, worker protocol helper and definition, CUDA adapter,
generator, rolling decoder, PNG smoke, runtime/generation/model-asset
preflights, bounded PNG validator, and streaming service. A recursive local
import-closure test plus mutation of the entrypoint and every companion
protects that inventory. `stack_sha256` remains the verified runtime bootstrap identity and
redundantly binds its CUDA guard bundle and exact runtime/model provenance. The
child rehashes the full worker bundle after runtime bootstrap and sends no
`HELLO` unless the observed stack identity equals the caller's frozen
expectation.

## Exact work and lifecycle contract

`START` accepts a nonempty UTF-8 prompt, an unsigned 32-bit seed, and exactly 21
latent blocks. It constructs a fresh `CF1LatentPullSession` and
`RollingTaehvChunkDecoder` without pulling work. For indices 0 through 20, one
credit causes one `session.pull()` followed by one `decoder.decode()`; the
released topology must be `[1] + [4] * 20`, every payload must be immutable PNG
bytes, and the worker cannot prefetch. The shared bounded PNG parser rejects
metadata/ancillary chunks and requires exact 832×480 RGB8 at
the CUDA session, serial smoke, service, and acceptance boundaries. Terminal
index 21 calls decoder
`finish()` before generator `finish()`, verifies both completed, emits exact
hash-bound evidence, and discards only the per-job objects.

Warm admission is checked again atomically when an async iterator claims the
worker. This prevents a stream object created while warm from secretly spawning
after cancellation returned the backend to cold state. Starting, stopping,
closed, poisoned, and dead-idle states fail closed. Cancellation and
`GeneratorExit` synchronously kill the child process group and reap it before a
future explicit warm; protocol failures and every other `BaseException`,
including `KeyboardInterrupt` and `SystemExit`, kill/reap and permanently poison
reuse. A clean `COMPLETE` is the only warm-reuse path.

The existing fake worker and browser demo remain unchanged and default. A
separate opt-in CF++1 browser entrypoint now connects this backend without
silently changing the fake demo. Its real browser acceptance now passes.

## Acceptance evidence boundary

The runner refuses an existing/symlink output, invalid/distinct-seed request,
noncanonical expected digest, or current worker-bundle drift before model boot.
The externally reviewed worker digest is never inferred as authorization. The
accepted complete worker execution closure, including the acceptance-only
fault hook and later diagnostics/guard hardening, hashes to
`59127c702b5c7cd394e7f9921e22d09b4ce05d2e4ef14dbf2d4b56ab64228fc8`.
The manifest binds that externally supplied expectation rather than adopting
whatever bytes happen to be present.

Per job the report retains the prompt hash, seed, PID/instance, stack and worker
digests, exact 21-chunk/81-frame topology, child/evaluator-reconciled ordered PNG
hashes, and integer parent-readiness times. It retains no prompt text, payload,
stderr, credential, auth header, or resolved environment value. Its explicit
fences authorize no quality, performance, browser-visible, provider-upload, or
additional-GPU claim. The emitter is a strict state machine: one first start,
contiguous pre-terminal chunks, exact topology before completion, terminal
summary equality, and no unsolicited, duplicate, or post-terminal event.

Publication is cooperative and `dirfd`-relative. It writes and `fsync`s one
private regular inode, hard-links the final name without replacement, verifies
the no-follow inode and exact bytes, removes only the staging name, `fsync`s
the directory, and reverifies the final name. The final name is never rolled
back. Any failure after the hard link reports `publication_indeterminate`
rather than a clean refusal, so the runner never claims that a potentially
visible accepted record was absent and never risks unlinking a competing inode.

## Review and verification

The actual-source consensus parent `ses_0801232baffelv15xHMyJan2Js` requested
`claude-opus,kimi-k3,grok` and timed out. Concrete Fable-5 child
`ses_0800e7723ffeRAnNbJUa3SzwYM` and Kimi-K3 child
`ses_0800e7720ffeCTHuE64tfS8QFY` produced zero output. Grok-4.5 child
`ses_0800e7720ffcPG163Kp5VMhIcp` retained partial reasoning but no final. No
Fable rate-limit signal occurred, so automatic Opus-4.8 fallback did not apply.
The recorded review cost is **$0**; this is not a completed panel verdict.

The acceptance-runner actual-source parent `ses_07fd42a29ffeFpp88gRx25wAye`
also timed out. It concretely selected Fable-5 first
(`ses_07fd009ffffeVr3ijTMpT5Yomt`), Kimi-K3
(`ses_07fd009fdffe4VImZju5mHuzyM`), and Grok-4.5
(`ses_07fd009fdffcVCXGsacbn0yO5j`). Fable and Kimi produced zero tokens; no
Fable rate-limit occurred, so Opus-4.8 fallback did not apply. Grok completed
NO-GO at **$0.180806**. Its principal claim that quick dead-worker failure does
not poison the registry was rejected against the exact source interval omitted
from its prompt: `WorkerProtocolError` reaches `next_chunk` as
`BackendFatalError`, which invokes the registry poisoner. The real-registry
regression proves the subsequent synchronous rejection. Grok's valid evidence
point was adopted: require `job_failed/backend_fatal` explicitly and derive the
reported registry poison from the observed rejected fourth start.

Independent actual-file audits then reproduced warm-state, dead-child,
base-exception, mutable-launch, environment-mutation, package-shadow, and forged
bytecode failures. A final adversarial pass additionally proved that the first
version only self-measured worker bytes after imports and left validated
timeouts/memory bounds mutable. Mandatory external bundle authorization,
pre-`Popen` plus pre-import checks, and complete operational-field locking were
added red-first. Each finding received a red regression before its fix. The final
complete-file audit is GO for this local non-authorizing scaffold with no
remaining P0/P1 finding. It explicitly withholds GO for real CUDA execution.

Session-16 adversarial audits independently checked the completed event-state,
PNG, bundle-closure, and publication changes and reported local GO with no
remaining P0/P1 finding. Later H100 execution exposed one additional real
ordering defect: sensitive-looking environment names were inspected after
model imports, so a benign library-added control could be misreported as having
crossed `exec`. The child now snapshots names before importing
`cf_streaming_worker` or any model/runtime module and carries that immutable
pre-model snapshot into `HELLO`. A red-first source-order regression proves the
old implementation fails and the new order passes.

The latest diff-fed parent `ses_07eba6ca6ffeZao49EpwVikquA` requested exactly
`claude-opus,kimi-k3,grok` and timed out. Fable child
`ses_07eb96845ffejMINZII6AmS8XV` produced zero output/tokens and no rate-limit
signal, so automatic Opus-4.8 fallback correctly did not run. Kimi child
`ses_07eb96842ffeDnxSt5CNEmht9B` and Grok child
`ses_07eb96841ffelPmwu896DGqDib` both returned GO for one bounded worker
acceptance and, only after it passed, one localhost browser acceptance. Their
costs total **$0.417366**. This is two recovered GO positions rather than a
synthesized three-voter verdict. The 80 affected acceptance/worker/process
tests pass after the environment-snapshot correction, and independent post-fix
review is GO for the development lifecycle boundary.

## Accepted H100 lifecycle and browser presentation

The planned H100 sequence completed: RunPod control, capture/reconcile/freeze,
unchanged admission, one-block smoke, full 21-block/81-PNG smoke, deterministic
video assembly/full decode, one exact-byte upload to both providers, then
`scripts/cf-streaming-acceptance` with two fixed seeds.

Manifest `30bfe91b…` records warm duration 517.196 seconds, two 21-chunk/81-frame
jobs on one PID/instance with distinct outputs, and the awaited forced-death
poison/reap proof. Its per-job serial PNG-readiness timings are diagnostics;
`performance_gate_evaluated` and every claim-authorization flag are false.

The fake browser remains the default. The separate opt-in real CF++1 runner now
has one accepted localhost result: genuine CF++1 H100/frozen-runtime labeling,
`streamState=complete`, 81 rendered frames, 21 chunks, 21 post-paint ACKs,
server completion, a visible 832×480 canvas showing the generated fox, and zero
console errors. The first acceptance instance shut down cleanly. A fresh warm
instance is intentionally live at `http://127.0.0.1:8765/` for the user's
manual test; the pod stays RUNNING until they finish. Filesystem output, server
emission, and browser completion still do not authorize performance.
