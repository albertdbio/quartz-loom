---
type: handoff
status: active
session: 9
date: 2026-07-20
description: "Restored and byte-verified the complete minimal CF++1/TAEHV/Wan-text stack, hardened its source inventory, and added a structurally tested direct CUDA bootstrap that avoids the stock diffusion and VAE weights."
branch: main
key_commits: []
prior_handoff: "session-8-persistent-process-worker"
---

# Session 9 Handoff — Pinned assets and CUDA bootstrap

## TL;DR

- The exact minimal CF++1 stack is locally complete. Source commit and all 11
  registered assets pass the no-follow, same-fd, mutation-aware verifier;
  `scripts/model-asset-preflight` returns `ready: true`.
- `bench/cf_cuda_adapter.py` directly constructs and strictly loads the CF++1
  generator and Wan text encoder, injects a latent passthrough, and constructs
  TAEHV separately. It avoids roughly 6.18 GB of unnecessary stock Wan
  diffusion/VAE weights.
- This is a structural bootstrap only. No CUDA module was imported locally, no
  tensor checkpoint was actually loaded by PyTorch, no frame was generated,
  and the persistent process/browser path still uses fake images.

## What changed

- Added the immutable model-asset lock, strict CLI preflight, and 11 red-first
  asset tests covering lock ambiguity, dirty/untracked/ignored injection,
  inherited Git redirection, symlink/FIFO rejection, size/hash errors, and
  mutation during hashing.
- Restored the clean pinned source, CF++1 EMA checkpoint, TAEHV implementation
  and weight, Wan model config, UMT5 text encoder, and tokenizer files under the
  ignored `.upstream/Causal-Forcing` cache.
- Added a seven-test CUDA bootstrap seam with deferred imports, exact model and
  config validation, safe mmap/assign loads, strict key equality, direct
  generator/text injection, latent passthrough, and a complete post-load asset
  recheck.
- Hardened the existing process seam to 35 tests: false isolation and sensitive
  environment claims are executable failures; protocol constants are
  single-sourced under `python -I`; HELLO equality-fences the latent cap; and
  the worker code digest now binds both entrypoint and protocol-definition bytes.

## Review

- Earlier panel parent `ses_0812d3743ffeNqf9A62rBV21dO` timed out. Fable
  child `ses_0812a1638ffe4F1u90mEwNuRj1` had no valid verdict, Kimi had no
  usable output, and Grok child `ses_0812a1633ffeOG052QDWsU6o8w` returned
  GO-WITH-FIXES. Its reproducible symlink/TOCTOU and latent-cap findings were
  closed.

- Final asset/process panel parent `ses_0811ac99dffeQeMEbUc0f1dk0N` timed out.
  Direct recovery found a concrete Fable-5 GO-WITH-FIXES final in
  `ses_08116fde9ffeeLTdM3BXbn8FM7`, a concrete Grok-4.5 GO-WITH-FIXES final in
  `ses_08116fde7ffejKAMoyuca0BwSI`, and no output in Kimi-K3 session
  `ses_08116fde8ffeg5L527sIUn1XC1`. All reproducible in-scope findings were
  closed red-first.
- A local Claude CLI `--model opus` fallback reviewed the exact bootstrap plus
  pinned upstream constructors and returned GO-WITH-FIXES. It independently
  confirmed the stock diffusion/VAE bypass. Its safe-load, config, tokenizer,
  and memory findings were incorporated; runtime CUDA compatibility remains
  explicitly unproved.
- The project-level gate remains NO-GO. Neither partial review clears evaluator,
  sustained-runtime, evidence, or real-worker requirements.

## Final observed verification

The root task reran these gates after the documentation update. The restarted
shell no longer exposed the old `python3.9` command and its system Python lacked
the pinned WebSocket dependency, so the complete suite was run under Python
3.9.6 in an ephemeral `uv` environment with the repository's exact
`aiohttp==3.13.5` requirement. No project or global dependency state changed.

- Complete Python 3.9 suite: **236/236**.
- Persistent-process suite: **35/35**.
- Process + service + NDJSON + WebSocket: **113/113**.
- Asset preflight: **11/11**; CUDA bootstrap: **7/7**.
- Executable Node browser client: **4/4**.
- Real asset gate: ready, exact source commit, zero unexpected paths, all 11
  assets verified.
- Python compilation and the model lock JSON parse pass.
- No GPU or paid API call was made, no secret was printed/persisted, and no
  commit was created.

## Next moves

1. Implement a dedicated real CUDA worker hosting a persistent
   `CF1StreamingSession` from the recovered loop and preserve exact
   `[1, 4 × 20]` dynamic TAEHV trimming, including fp16-before-TAEHV.
2. Add synchronized CUDA-event → D2H → production raster ownership before each
   protocol `CHUNK`, then replace only the fake worker core.
3. Bind the observed canonical effective-config digest, adapter, protocol, and
   assets into one worker-derived stack identity. Run a bounded real GPU boot
   smoke before any generation sweep.
4. Separately complete human-anchored Gemini calibration and resolve the
   impossible 241-latent/global-cache contract before confirmatory generation.

## See also

- [[Pinned-CF1-CUDA-Bootstrap]]
- [[Streaming-Service-Boundary]]
- [[State]]
- Prior: [[session-8-persistent-process-worker]]
