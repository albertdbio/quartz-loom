---
type: handoff
status: active
session: 15
date: 2026-07-20
description: "Added a production-shaped, non-authorizing persistent-H100 lifecycle acceptance CLI with real registry/emitter coverage, independent payload evidence, safe forced-death proof, and durable no-clobber publication; local tests are GO while the candidate runtime still refuses before HELLO."
branch: main
key_commits: []
prior_handoff: "session-14-persistent-cuda-worker"
superseded_by: "session-16-acceptance-evidence-hardening"
---

# Session 15 Handoff — persistent-worker H100 acceptance

## TL;DR

- Added `bench/cf_worker_acceptance.py` and
  `scripts/cf-streaming-acceptance`. The command prewarms the exact real worker,
  runs two sequential 21-chunk/81-PNG jobs through the actual
  `StreamingJobRegistry`, independently hashes transient emitted bytes, and
  reconciles them with child `COMPLETE` evidence.
- Added an identity-fenced backend fault hook that verifies the idle child owns
  `PID == PGID == SID`, sends `SIGKILL` to that exact process group under the
  lifecycle lock, then deliberately leaves stale state for the next ordinary
  registry job to detect, poison, and reap.
- Success publishes one fsynced, atomic hard-link/no-replace JSON manifest. The
  report retains hashes and integer timing metadata, never prompt text, frames,
  stderr, environment values, or credentials. It explicitly authorizes no
  quality, performance, browser, provider-upload, or further-GPU claim.
- The checked-in runtime remains `candidate`, so real acceptance still refuses
  before `HELLO`. No H100, model, provider API, or upload ran; no commit was
  created.

## Frozen inputs and exact command

The worker bundle changed when the acceptance-only fault hook joined the real
worker wrapper. The session-15 snapshot was
`cdc648d6a723415e1be5b59517805bbe114befa31116b7056c1d2e4ebe5ddf4f`; it is
superseded and must not be used for a new launch. Session 16 expanded the
pre-import bundle to the complete local execution closure. The current
externally reviewable candidate is:

```text
b343acac453c066e0c00fbaa8b7b52de2f0bfb180f893c228b50e337830c2cd3
```

Do not silently recompute and adopt that value on the H100. Re-review/freeze it
beside the runtime, then pass both external expectations unchanged:

```bash
./scripts/cf-streaming-acceptance \
  --prompt '<exact public generation prompt>' \
  --first-seed 11 \
  --second-seed 12 \
  --expected-stack-sha256 '<reviewed runtime bootstrap identity>' \
  --expected-worker-code-sha256 b343acac453c066e0c00fbaa8b7b52de2f0bfb180f893c228b50e337830c2cd3 \
  --runtime-image-digest 'sha256:<provider-observed immutable image digest>' \
  --output '/absolute/new/sibling/worker-acceptance.json'
```

The output and its parent must already satisfy the CLI's no-clobber boundary.
Do not add timeout, worker-path, environment, skip, retry, or auto-upload flags.

## What the success record proves

Each accepted job must retain the same live PID and worker instance, exact stack
and worker digests, distinct unsigned-32-bit seed, `[1] + [4] * 20` topology,
81 immutable PNG hashes, and a matching service summary. The emitter hashes the
actual payloads received at the service boundary and discards the bytes; child
completion hashes must match that independent ordered vector exactly.

After both jobs, the backend-owned fault hook signals only the verified idle
worker group. The runner does not poll the worker between that signal and the
third awaited registry job. That job must emit `job_failed/backend_fatal`; the
ordinary supervisor path must permanently poison, reap, and clear the child;
the registry must then reject a fourth `start` synchronously. Only after backend
`close()` succeeds is the manifest published.

Timing is deliberately bounded development evidence: warm duration is separate;
job timings describe parent-received, validated PNG readiness through generation,
decode, D2H, PNG encoding, IPC, and service validation. They exclude MP4,
network, upload, browser delivery, paint, and sustained throughput.

## Review and verification

The requested actual-source consensus parent is
`ses_07fd42a29ffeFpp88gRx25wAye`. It concretely selected Fable-5 child
`ses_07fd009ffffeVr3ijTMpT5Yomt` first, Kimi-K3 child
`ses_07fd009fdffe4VImZju5mHuzyM`, and Grok-4.5 child
`ses_07fd009fdffcVCXGsacbn0yO5j`, then the bridge timed out. Fable and Kimi
produced zero tokens. No Fable rate-limit was recorded, so the configured
Opus-4.8 fallback did not apply. Grok completed NO-GO at **$0.180806**.

Grok's primary poison-coupling P0 was rejected against the exact omitted source:
dead-worker `WorkerProtocolError` is a `BackendFatalError`; the backend
poisons/reaps it and `run_stream_job.next_chunk` calls the registry poisoner,
emitting `backend_fatal`. The real-registry regression proves the fourth start
is rejected. One evidence-hardening point was adopted: the runner now requires
that exact failure code and derives `registry_poisoned` from the observed
synchronous rejection instead of hardcoding it.

- Focused acceptance/worker/process/service: **101/101**.
- Dependency-complete Python discovery: **362/362**.
- Node browser client: **4/4**.
- `compileall` and the real CLI help surface: green.

## Exact next sequence

1. Restore authenticated RunPod control; capture, reconcile, independently
   review, and freeze the immutable H100 runtime plus worker digest.
2. Pass the unchanged authorizer and run one block; inspect the PNG and exit.
3. Run 21 blocks; assemble and credential-free preflight the exact MP4.
4. Explicitly upload those same bytes once through `cf-video-understand upload`;
   Gemini runs first and TwelveLabs/Pegasus complements it. Never automatically
   retry an `in_flight`/`uncertain` provider.
5. Run the acceptance command above. Only its success may unlock real-backend
   browser binding; it still does not unlock a performance or quality claim.

## See also

- [[CF1-Persistent-CUDA-Worker]]
- [[CF1-Development-Video-Artifact]]
- [[CF1-H100-Runtime-Preflight]]
- [[State]]
- Superseding hardening: [[session-16-acceptance-evidence-hardening]]
- Prior: [[session-14-persistent-cuda-worker]]
