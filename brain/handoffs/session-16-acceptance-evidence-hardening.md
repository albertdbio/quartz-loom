---
type: handoff
status: active
session: 16
date: 2026-07-20
description: "Closed the acceptance-evidence state-machine, PNG-validation, worker-bundle, and publication-integrity gaps; all local audits and tests are GO, while consensus and authenticated H100 execution remain pending."
branch: main
key_commits: []
prior_handoff: "session-15-persistent-worker-acceptance"
superseded_by: "session-17-frozen-h100-generation-and-browser-acceptance"
---

# Session 16 Handoff — acceptance-evidence hardening

## TL;DR

- The acceptance emitter is now a strict state machine. `job_started` must be
  first and unique; chunks must be pre-terminal and contiguous; successful
  completion requires the exact 21-chunk/81-frame topology and the terminal
  event's `StreamSummary` must equal the task result; unsolicited job IDs,
  duplicate/late events, and malformed failure/cancellation terminals refuse.
- Every CF PNG boundary now uses the bounded, metadata-free PNG subset and
  requires exact 832×480 RGB8 output. The service, CUDA
  session, serial smoke, and acceptance harness therefore share one strict
  renderable-byte contract rather than trusting signatures or dimensions
  reported elsewhere.
- The externally frozen worker digest now covers the complete 15-file local
  execution closure before the child imports project code: the entrypoint,
  protocol definition, package initializer, process/worker protocol code,
  CUDA adapter/generator/session/smoke, runtime/generation/asset preflights,
  PNG validator, and streaming service. Recursive-import plus entrypoint and
  companion-mutation regressions protect that inventory.
- Success publication is a cooperative `dirfd`-relative no-clobber protocol:
  stage and `fsync` one private regular inode, hard-link without replacement,
  verify no-follow inode and exact bytes, unlink only the staging name, `fsync`
  the directory, and reverify the final name. Every failure after the hard link
  is `publication_indeterminate`; the publisher never rolls the final name
  back or risks deleting a competing inode.
- Local implementation audits are GO. Relevant acceptance/worker/process/
  service/PNG tests pass **116/116**, the dependency-complete Python suite
  passes **379/379**, and the Node client suite passes **4/4**. The requested
  post-hardening consensus retry is still pending and cannot be inferred from
  the local audits.

## Current reviewed worker-bundle candidate

The digest was recomputed from the current tree with the repository's canonical
`worker_bundle_sha256` mechanism—not with an ad hoc shell hash:

```bash
task_pycache=$(mktemp -d /tmp/realtime-video-brain-pycache.XXXXXX)
PYTHONPYCACHEPREFIX="$task_pycache" python3 - <<'PY'
from pathlib import Path
from bench import cf_streaming_process_worker, cf_streaming_worker
from bench.streaming_process_protocol import worker_bundle_sha256

print(worker_bundle_sha256(
    Path(cf_streaming_process_worker.__file__).resolve(),
    cf_streaming_worker.REAL_WORKER_BUNDLE_PATHS,
))
PY
```

Current output:

```text
b343acac453c066e0c00fbaa8b7b52de2f0bfb180f893c228b50e337830c2cd3
```

This supersedes the session-15 snapshot digest. It is still a review candidate,
not permission to self-adopt observed H100 bytes: independently review and
freeze it beside the immutable runtime identity before launch.

## Operational boundary

- No GPU/model execution, video-understanding API call, or upload ran during
  this hardening increment.
- The local untracked `.env` contains the variable names `GEMINI_API_KEY` and
  `TWELVELABS_API_KEY`. Their values were not read, printed, persisted, or sent
  to any reviewer.
- Authenticated RunPod control and availability of the dedicated H100 have not
  yet been re-verified. Do not infer either from the prior browser tab or the
  stopped pod record.
- Only a fresh artifact assembled from the exact new 21-block H100 run may
  enter video understanding. Run the credential-free preflight first, then
  make one explicit upload of those exact bytes to both Gemini and TwelveLabs.
  Do not upload any legacy artifact. Do not automatically retry a provider in
  `in_flight` or `uncertain` state.

## Exact next sequence

1. Restore and verify authenticated RunPod control and the dedicated H100.
2. Capture, reconcile, independently review, and freeze the immutable runtime,
   stack identity, OCI image digest, and worker digest above.
3. Pass the unchanged authorizer; run the one-block process-to-exit smoke, then
   the complete 21-block/81-PNG smoke.
4. Assemble only those fresh PNGs into the exact deterministic MP4 and run
   `scripts/cf-video-understand preflight` without credentials.
5. After that preflight succeeds, invoke one explicit `upload` for those same
   bytes to both providers. Stop on `in_flight` or `uncertain`; do not retry.
6. Run `scripts/cf-streaming-acceptance` with two distinct fixed seeds, the
   frozen identities, and a new sibling output. Its local hardening does not
   substitute for the still-pending consensus retry or real H100 evidence.

## See also

- [[CF1-Persistent-CUDA-Worker]]
- [[CF1-Development-Video-Artifact]]
- [[CF1-H100-Runtime-Preflight]]
- [[State]]
- Prior: [[session-15-persistent-worker-acceptance]]
- Next: [[session-17-frozen-h100-generation-and-browser-acceptance]]
