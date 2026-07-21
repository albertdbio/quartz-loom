---
type: gotcha
status: active
date: 2026-07-19
description: "Matching prompt and seed do not prove two generated videos are the same artifact; bind the effective config, initial noise, latent, decoder path, and encoded-media hashes."
anchors: ["bench/quality_sweep.py#validate_run_manifest", "bench/quality/quality-repair-v1.protocol.json"]
related: ["[[ADR-001-quality-qualified-headline]]", "[[State]]"]
---

# Gotcha — Seed equality is not artifact provenance

The archived performance and quality files reused prompt/seed labels, which
made them look paired. They were not the same generated artifacts: the
performance path used rolling three-latent TAEHV while the quality videos used
one full-batch TAEHV call, and their encoded bytes differ substantially.

A seed only initializes a stochastic path. Code revision, effective config,
RNG algorithm/device/application point, input noise, intermediate latent,
decoder execution, and encoder command can all change the output under the
same integer seed.

For every registered run, retain and validate all of those pins plus
`initial_noise_sha256`, `input_noise_sha256`, `latent_sha256`, and
`media_sha256`. Never join performance and quality evidence on prompt/seed
alone.
