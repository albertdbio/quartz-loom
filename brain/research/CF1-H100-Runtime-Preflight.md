---
type: research
status: active
date: 2026-07-20
description: "The CF++1 H100 path has a frozen schema-v2 runtime/evidence lock and exact OCI/Python/package/native identities. A later relaunch correctly refused 62 generated .pyc files totaling 1,006,933 bytes despite unchanged packages/source; exact cache-only removal restored lock equality and the no-bytecode wrapper relaunched cleanly. All execution and timings remain non-authorizing."
anchors: ["bench/cf_runtime_preflight.py#preflight_current_runtime", "bench/cf_runtime_preflight.py#runtime_capture_report", "bench/cf_runtime_preflight.py#capture_runtime_environment", "bench/cf_runtime_preflight.py#validate_current_host_capacity", "bench/cf_runtime_preflight.py#validate_loaded_cuda_capacity", "bench/cf_runtime_preflight.py#validate_cf1_tokenizer_sentinel", "bench/cf_cuda_adapter.py#build_cf1_runtime", "bench/cf_cuda_session.py#_require_verified_runtime"]
related: ["[[Research]]", "[[State]]", "[[Pinned-CF1-CUDA-Bootstrap]]", "[[CF1-Rolling-TAEHV-Session]]", "[[CF1-Latent-Pull-and-Smoke]]", "[[session-11-runtime-preflight]]", "[[session-12-latent-pull-and-smoke]]", "[[session-18-motion-diagnosis-and-looping-replay]]"]
---

# CF++1 H100 runtime preflight

## Bottom line

`bench/runtime/cf1-h100-cu128-v1.lock.json` and the stdlib-only
`bench/cf_runtime_preflight.py` define the admission boundary that passes before
the CF++1 adapter reads the 17 GiB asset set or constructs the model. The
schema-v2 lock is now `frozen`, has no unresolved fields, and hashes to
`d4d163d635ecbafb5b11bbe54cca7bdd5e9f80c1edc23ed96972821940ecc692`.
Its bound evidence file hashes to
`8209043b4ebecc85f0e844f9c040b54fc1685104fe9e0b361ce9ee6d060b0c6c`.

The unchanged authorizer passed on the dedicated H100. It admitted CPython
3.12.3, CUDA 12.8, Torch 2.8.0+cu128, driver 580.126.09, one NVIDIA H100 80GB
HBM3, and actual FlashAttention-2 under the provider-observed OCI index, child
manifest, and config digests. The locked runtime-environment identity is
`858c16004d327fab276c8a8e43aa61bfdfcdbe78033e82edd96dfbe434839b7c`,
native identity is
`8884c4be14c4c7c404b346a76347ebd3ed8218c229b6066f4ceba743108c619c`,
and the bound loaded-runtime probe identity is
`070efce1bdcd5972fd9cdde7f02d66dab3320a568e7fb04caa2575aae8f0dd4c`.
Real one-block/full smoke and persistent-worker lifecycle execution have now
passed behind this boundary. That proves admission and compatibility, not
performance, quality, sustained operation, or browser presentation.

## Frozen lock and admission provenance

The frozen lock resolves the prior image, Python, driver, distribution,
wheel/native, CUDA-ABI, and backend blockers. It retains the complete normalized
75-distribution installed set and binds the evidence file rather than trusting
a reconstructed version list. Admission requires exact equality for immutable
OCI child digest, Python version/build, packages, driver, platform, GPU name and
capability. The OCI assertion remains provider-observed deployment metadata,
not in-container attestation and not a desired-value echo.

The environment identity hashes stable runtime facts and the exact lock. It
does not hash transient free-memory readings or copy the desired attention
backend into an observation. Capacity is retained separately, and the actual
backend selected after upstream import is recorded in bootstrap provenance.

## Fail-closed generated-bytecode drift

The synchronized replay-client relaunch did not silently reuse a changed
runtime tree. Static evidence refused admission because exactly 62 generated
`.pyc` files totaling 1,006,933 bytes had appeared across aiohttp and its
dependencies. The installed package versions and source bytes still matched;
the entire observed difference was the enumerated cache set.

Cleanup proceeded only after validating every affected path, the exact file
count, and the exact aggregate byte count. Only those 62 cache files were
removed. The complete static-evidence verifier then returned to exact equality
with the frozen lock; neither the evidence nor lock was recaptured or relaxed.
The launch wrapper now carries the no-bytecode guard. The clean relaunch passed
the same unchanged admission boundary and then passed exact real-browser replay
reacceptance.

This is a successful refusal/recovery proof, not permission to normalize future
drift. Any later mismatch must again be identified exactly before a bounded
remediation; package, source, native-library, or unclassified changes remain a
hard stop.

## Capacity contract

The pinned direct-load path has these byte-accounting anchors:

| Quantity | Bytes |
|---|---:|
| Locked asset files | 17,082,290,819 |
| Known active host storage | 39,761,449,216 |
| Conservative host no-reclamation envelope | 48,364,777,331 |
| GPU weight storage | 14,222,445,350 |
| GPU cache-ready lower bound | 20,355,140,390 |
| Archived rolling peak allocated / reserved | 24,822,243,328 / 29,360,128,000 |
| Archived full-batch peak allocated / reserved | 38,896,813,056 / 50,545,557,504 |

Admission requires at least 56 GiB effective host headroom, at least
80,000,000,000 GPU bytes total, and at least 36 GiB GPU bytes free. Effective
host headroom is the minimum of `/proc/meminfo` `MemAvailable` and every finite
current-to-root cgroup memory headroom. Both cgroup v2 and v1 membership are
handled; swap never counts. Exactly one visible GPU is allowed and it must be
the H100 at logical `cuda:0`.

Host capacity is checked before imports and again after asset verification and
third-party imports, immediately before constructors. Torch then rechecks its
loaded version/CUDA runtime, selected device name/capability, independent free
and total memory, and the actual selected attention backend. These thresholds
authorize only the rolling path. They do not authorize the archived full-batch
decoder. The real rolling-path boot and 517.196-second warm completed inside the
admitted envelope; that proves fit for this bounded run, not a performance-
authorized peak-memory benchmark.

## Tokenizer sentinel

Before any checkpoint load or model construction, the exact tokenizer instance
that will be injected into the text encoder must tokenize these raw prompts at
length 512 with the pinned whitespace cleaner:

```text
"  A  red\tfox\njumps.  "
"Café 猫"
"<extra_id_0>"
```

The non-padding ID prefixes are:

```text
[320, 4062, 273, 56209, 48150, 281, 274, 1]
[25382, 273, 14985, 1]
[256299, 1]
```

The contract also pins shape `3×512`, CPU int64 tensors, mask sums `8/4/2`,
contiguous masks, right-padding ID 0, EOS 1, unknown 3, `<extra_id_0>` 256299,
and vocabulary size 256300. The versioned binary framing of IDs and masks has
SHA-256 `2ab00c08615e582d62b163a0d13d305c04c3f3c99a45e034483413a7efb2210f`.
This sentinel detects important tokenizer drift but does not replace the exact
dependency and file pins.

## Adapter and session provenance

`build_cf1_runtime` performs runtime admission before asset hashing, then
rechecks host and loaded CUDA capacity before constructors. The resulting
runtime carries the exact Torch module object and the backend classified from
the imported upstream attention flags (`flash-attention-3`, then
`flash-attention-2`). The pinned cross-attention implementation has no operable
SDPA fallback and bootstrap refuses if neither FlashAttention backend is
available. Bootstrap provenance binds the actual backend plus runtime-lock,
stable environment, tokenizer, config, asset, and six-module guard identities.

`RollingTaehvChunkDecoder` accepts only logical `cuda:0`, requires the
provenance backend to match the runtime backend, and requires the caller to pass
the same Torch module object that the verified runtime used. This closes two
reviewed substitution paths: copying the lock's desired backend into identity,
and handing the decoder an unrelated Torch-like object.

## Review history and completed freeze

Two scrubbed actual-source consensus calls requested
`claude-opus,kimi-k3,grok`:

- parent `ses_080a41455ffebwcKeuxBusoFCV`: only concrete `grok-4.5` child
  `ses_0809f9bc2ffezqjMXterCjImqL` completed, at local GO-WITH-FIXES / project
  NO-GO; Fable-5 and Kimi-K3 recorded zero output;
- parent `ses_080991858ffe6xDDU2VSlFUNW9`: only concrete `grok-4.5` child
  `ses_08092f5bfffeOis8yPeUlO7Z9o` completed, at local NO-GO / project NO-GO;
  Fable-5 and Kimi-K3 again recorded zero output.

Neither call exposed a detected Fable rate limit or automatic Opus fallback.
Direct SQLite recovery retained the completed Grok analyses. Their reproducible
free/total accounting, OCI formatting, package-inventory, and logical-GPU
findings were fixed red-first; excerpt/truncation claims were checked against
the complete files and rejected when they did not reproduce.

Because both panels were incomplete, a separate local Claude CLI review of the
complete source requested Opus and exposed concrete `claude-opus-4-8`, session
`74295a60-90f9-4ce9-9084-7c670774b2c0`. It returned local NO-GO / project
NO-GO. Its actual-backend, exact-Torch-binding, transient-capacity identity, and
structured-exception findings are fixed. Its full-package-inventory concern was
retained as a freeze blocker rather than relaxed. The two
Grok votes cost $0.289212 and the local Claude review cost $1.157491, for
$1.446703 total external review spend.

A final independent read of the complete on-disk implementation found one more
real issue: the classifier called the no-FlashAttention state `torch-sdpa`, but
the pinned upstream cross-attention calls `flash_attention` directly and
asserts FA2 in its non-FA3 branch. Red-first tests now prove that both the lock
and runtime refuse when neither FA2 nor FA3 is available. The post-fix audit
reports local GO with no remaining in-scope finding.

The later dedicated-worker capture supplied the previously missing OCI, exact
Python build, complete distribution/tree, native-library/CUDA-ABI, driver, and
loaded-backend evidence. Independent reconciliation produced the frozen
schema-v2 lock with `unresolved: []`; the unchanged authorizer then passed.
Capture itself remains non-authorizing and cannot edit the lock, but the
reviewed frozen lock now legitimately authorizes the exact admitted boot. The
project-level quality, performance, and browser gates remain separate and open.

## Current result

The frozen authorizer passed before the real one-block smoke, full
21-block/81-PNG smoke, and accepted persistent worker. The accepted worker
manifest is SHA-256
`30bfe91b4742bbd7e04966f9baed75562fce3c0a501b05c0063a45b7c54de115`;
it binds current stack identity `349c79f4…`, worker identity `59127c70…`, and
reports no sensitive launch-environment names. Warm took 517.196 seconds, but
the manifest explicitly does not evaluate or authorize performance.

The original localhost real-CF++1 browser acceptance passed: the genuine opt-in
H100/frozen-runtime page reached 81 painted frames, 21 chunks, 21 post-paint
ACKs, server completion, a visible canvas, and zero console errors. The first
acceptance instance shut down cleanly. The later replay-client relaunch first
failed on the exact 62-file drift above, returned to exact frozen evidence only
after cache-only remediation, passed the frozen admission boundary under the
no-bytecode guard, and then passed exact synchronized-build browser replay
acceptance at 81 frames / 21 chunks / 21 ACKs with counter-neutral visible
replay and zero console errors.
Neither presentation proof nor drift recovery authorizes quality or
performance.
