---
type: handoff
status: active
session: 14
date: 2026-07-20
description: "Added and adversarially hardened a warm-only provenance-bound CF++1 process worker; local code is GO, while the candidate runtime still refuses before HELLO and real H100 acceptance remains pending."
branch: main
key_commits: []
prior_handoff: "session-13-development-video-artifact"
---

# Session 14 Handoff — persistent CUDA worker scaffold

## TL;DR

- Added [[CF1-Persistent-CUDA-Worker]]: a distinct real-worker entrypoint and
  backend around the verified runtime, exact latent pull, rolling TAEHV, and
  pinned PNG path.
- Extended the existing process supervisor with explicit prewarm, exact child
  environment/bundle controls, warm-only atomic admission, and fail-closed
  `BaseException` cleanup.
- The actual candidate-lock subprocess still exits before `HELLO`, as required.
  No CUDA/model/API/upload ran and no commit was created.

## What changed

`bench/cf_streaming_worker.py` owns one verified warm runtime and creates fresh
generator/decoder state for each exact 21-block job. One `NEXT` earns exactly
one pull and decode; terminal completion finalizes decoder then generator before
reuse. `bench/cf_streaming_process_worker.py` performs runtime construction,
identity checks, CUDA synchronization, and a post-bootstrap bundle rehash before
`HELLO`. A mandatory externally frozen worker digest is checked before
`Popen`, passed to the child, independently recomputed in a standard-library
prelude before project imports, and checked around bootstrap.

`bench/streaming_process.py` now supports explicit `warm()`, warm-required
admission, companion bundle hashing, and a copied allowlisted child environment.
Cancellation, generator close, protocol faults, interrupts, and system exits
retire the process safely. The CF backend's launch configuration and every
validated timeout, wire/payload/prompt/job limit, and stderr-retention bound are
immutable. The real child runs under `python -I`, uses a local regular `bench` package, and
redirects bytecode lookup before project imports so later package shadows or
forged timestamp-valid local `.pyc` files cannot bypass the checked source.

The fake worker and browser path are unchanged. The scaffold is intentionally
not wired into the browser before real H100 acceptance.

## Review and verification

Consensus parent `ses_0801232baffelv15xHMyJan2Js` timed out. Fable-5 child
`ses_0800e7723ffeRAnNbJUa3SzwYM` and Kimi-K3 child
`ses_0800e7720ffeCTHuE64tfS8QFY` had zero output; Grok-4.5 child
`ses_0800e7720ffcPG163Kp5VMhIcp` retained partial reasoning without a final.
There was no Fable rate limit, hence no Opus-4.8 fallback. Recorded cost is $0.

Independent actual-file audits supplied adversarial reproducers for stale warm
iterators, transitional and dead-idle states, `KeyboardInterrupt`/`SystemExit`,
mutable CF launch fields, post-construction secret environment insertion,
package shadowing, forged `.pyc` substitution, dynamic post-import bundle
self-acceptance, and mutable operational bounds. All failed red before their
fixes. Final verdict: GO with no remaining P0/P1 for the local non-authorizing
scaffold; not GO for real CUDA.

- Focused real-worker plus process suites: **56/56**.
- Dependency-complete Python discovery with temporary `aiohttp==3.13.5`:
  **353/353**.
- Bare Python: 333 passes plus the known missing-`aiohttp` WebSocket import.
- Node browser client: **4/4**; `compileall`: green.

## Operational state and exact next sequence

The stopped RunPod pod remains `ooprxl8l5c7c59`; the local browser console was
signed out at last check. Local Gemini/TwelveLabs keys and the newly enabled
video-upload permission were not used because no current H100 artifact exists.
Legacy MP4s remain excluded.

1. Restore authenticated RunPod control and capture the provider-observed H100
   image plus exact Python/package/wheel/ABI/driver/backend facts.
2. Reconcile, independently review, and freeze the runtime lock plus external
   worker-bundle digest; require both unchanged expectations to pass.
3. Run one-block serial smoke, inspect the exact PNG, and exit.
4. Run 21 blocks, assemble the exact MP4, credential-free preflight it, then
   explicitly upload the same bytes once to Gemini and TwelveLabs.
5. On the frozen H100, explicitly warm this worker, complete two jobs on one
   PID/runtime, then force EOF/failure and prove kill/reap/poison.
6. Only then connect the real backend to the browser and measure presentation.

## See also

- [[CF1-Persistent-CUDA-Worker]]
- [[CF1-H100-Runtime-Preflight]]
- [[CF1-Latent-Pull-and-Smoke]]
- [[CF1-Development-Video-Artifact]]
- [[State]]
- Prior: [[session-13-development-video-artifact]]
