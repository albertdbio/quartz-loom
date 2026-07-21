---
type: research
status: active
date: 2026-07-20
description: "TwelveLabs Pegasus 1.5 is transport-verified, but exact-byte comparisons show score inflation and unreliable temporal interpretation. It correctly returned 0/100 static/no-bounce for the terse CF++1 ball clip and recognized expanded-prompt motion, yet over-credited one seed with physically impossible timestamps. The adapter is useful for corroboration; the judge remains calibration-failed."
source_url: "https://docs.twelvelabs.io/v1.3/api-reference/analyze-videos/analyze"
anchors: ["bench/video_judge.py#build_pegasus_request", "bench/video_judge.py#parse_pegasus_response", "bench/video_judge.py#run_pegasus_plan"]
related: ["[[Research]]", "[[State]]", "[[Gemini-3.1-Video-Judge-Calibration]]", "[[CF1-Prompt-Conditioned-Motion]]", "[[ADR-001-quality-qualified-headline]]", "[[session-4-quality-protocol-hardening]]", "[[session-5-gemini-calibration-and-runner-preflight]]", "[[session-18-motion-diagnosis-and-looping-replay]]"]
---

# Pegasus 1.5 video-judge calibration

## Bottom line

The TwelveLabs integration is a **transport GO and judge-calibration NO-GO**.
Four real, bounded calls reached the official Pegasus 1.5 endpoint, returned
clean structured responses, and passed the strict local parser. The first
three were the calibration canaries below. A later Gemini 3.1 Pro calibration
judged those **same three media hashes** twice at an explicit 16 fps:

| System | Pegasus 1.5 canary mean | Gemini 16-fps two-pass mean | Offset |
|---|---:|---:|---:|
| CF1 | 9.0 | 5.7 | +3.3 |
| CF2 | 8.6 | 4.1 | +4.5 |
| SF4 | 8.0 | 5.6 | +2.4 |

Pegasus ranked **CF1 > CF2 > SF4**. Gemini's two-pass means rank
**CF1 > SF4 > CF2**, with CF1/SF4 separated by only 0.1 and swapping order
between passes. The decisive failure is absolute: Pegasus passes all three at
the 7/10 bar; Gemini passes none. Pegasus also describes stable motion/anatomy
where Gemini repeatedly identifies walking/sliding, morphing, or a static
subject. The canaries establish API and parser viability, not evaluator
validity. They are not development-selection or confirmatory gate evidence.

After upload permissions were adjusted, a fourth bounded call rechecked the
real upload path against `POST https://api.twelvelabs.io/v1.3/analyze` with
`pegasus1.5`. It completed with `finish_reason: stop`, reported 3,001 input and
151 output tokens, and returned scores of 9, 9, 9, 8, and 9 (core mean 8.8) for
media hash
`e70a25c99960099fcfcf6269817f14a9642a7af105eb1245c9619c92b1775351`.
The upload and strict parser succeeded. This is additional transport evidence,
not calibration evidence; the high score reinforces the existing inflation
finding, so calibration remains failed.

## Motion-diagnostic corroboration

The session-18 CF++1 motion matrix provides a useful discriminative check that
is narrower than the quality canaries. Pegasus independently returned 0/100
motion for the exact `short-seed7` clip and described the ball as static with
no bounce. That agrees with direct inspection and the measured 4 px vertical
ball-center span. It also recognized clear motion in the two expanded-prompt
clips, agreeing directionally with their measured 152 px and ≥220 px spans.

The same check reinforces the calibration boundary rather than relaxing it.
Pegasus over-credited one expanded seed and supported its physical narrative
with impossible timestamps when checked against the exact 81 frames. It can
distinguish gross static-versus-moving behavior here, but its temporal and
physical interpretation is not dependable. Treat it as independent
corroboration only; manifest-bound frames, deterministic metrics, and human
inspection must resolve disagreements. This result does not make Pegasus
quality-qualified and cannot enter the replacement gate.

The earlier comparison in this note used archived `*_p0.mp4` Gemini results.
Those files were not the Pegasus canary bytes, and the archived CLI calls also
used Gemini's default 1-fps sampling. That comparison is superseded by the
exact-byte 16-fps result above; see [[Gemini-3.1-Video-Judge-Calibration]].

Protocol readiness therefore remains `calibration-failed`. A model family must
be `quality-qualified` before protocol freeze or full-plan fan-out.

## Official provider contract used

- Exact model identifier: `pegasus1.5`.
- Endpoint: `POST https://api.twelvelabs.io/v1.3/analyze`.
- Mode: synchronous, non-streaming general analysis with a structured JSON
  response. This is the simplest contract for the 5.0625-second local videos;
  asynchronous tasks are unnecessary for these canaries.
- Local adapter transport: inline base64 video, below the documented 30 MB
  limit, `temperature: 0`, `stream: false`, and a documented-subset JSON Schema.
- Success acceptance: `finish_reason` must equal `stop`; `data` must parse as
  one JSON object with exactly the registered fields and integer scores in
  `[1, 10]`; `usage.output_tokens` must be present. Truncated, malformed,
  duplicate-key, extra-field, or out-of-range results fail closed.

Official references:

- [Analyze API reference](https://docs.twelvelabs.io/v1.3/api-reference/analyze-videos/analyze)
- [Structured responses](https://docs.twelvelabs.io/docs/guides/analyze-videos/structured-responses)
- [Pegasus 1.5 release notes](https://docs.twelvelabs.io/docs/get-started/release-notes)
- [Pegasus migration guide](https://docs.twelvelabs.io/v1.3/docs/get-started/migration-guide)
- [API error codes](https://docs.twelvelabs.io/v1.3/api-reference/error-codes)
- [Pricing](https://www.twelvelabs.io/pricing)

## Local implementation and evidence boundary

- `bench/video_judge.py` pins the endpoint/model, constructs the structured
  request, validates exact score fields, binds media/request hashes, writes
  resumable scrubbed evidence atomically, and excludes base64 media and API
  credentials from persisted evidence.
- `scripts/quality-video-judge` exposes a bounded `canary` path and a registered
  blind-plan path. The latter refuses to fan out a model family unless its
  protocol readiness is `quality-qualified`.
- `bench/schemas/quality-model-evidence-v1.schema.json` is the persisted
  evidence contract. It binds model, endpoint, blind asset, media hash, prompt
  and rubric hashes, request parameters, raw scrubbed response, and response
  hash while forbidding unexpected fields.
- `bench/tests/fixtures/twelvelabs-pegasus15-sync-response.json` is a captured,
  scrubbed successful response used to exercise the parser without another API
  round-trip.
- The provider credential exists only in the local untracked environment. Its
  name and value are not persisted in this note or request/evidence artifacts.

## Canary accounting

Each bounded call submitted a 5.0625-second fox video, for 20.25 submitted video
seconds total. The four responses used 685 output tokens. At the current
published developer rates—$0.0292 per analyzed video minute and $0.0075 per
1,000 output tokens—the estimated total is:

```text
(20.25 / 60) * $0.0292 + (685 / 1000) * $0.0075
= $0.0149925 ≈ $0.0150
```

The permission-recheck call alone is estimated at `$0.00359625`.

This is an estimate from submitted duration and returned output-token usage;
the response does not report a dollar charge.

## Calibration requirement before use

The two registered Pegasus passes are repeatability checks for one deterministic
judge configuration. They do not count as two independent model judgments;
independence comes from distinct evaluator families and the human panel.

Provider responses can be re-parsed from archived Pegasus evidence. Human
ratings instead remain process-attested: their hashes bind custody and prevent
later score drift, but do not cryptographically prove who supplied a judgment.

Keep the working transport and strict evidence path, but do not run the blind
gate tensor through Pegasus yet. Qualification needs a registered calibration
set with known pairwise distinctions, temporal-failure coverage, repeat passes,
and agreement thresholds against independent human and model-family judgments.
Only evidence that demonstrates stable ranking and useful separation can move
the family from `calibration-failed` to `quality-qualified`; otherwise replace
the family. Deterministic finalist-selection implementation and tests are
complete, but real registered Round-A/Round-B evidence remains a separate
unmet gate.
