---
type: research
status: active
date: 2026-07-20
description: "The first real 81-PNG CF++1 H100 smoke was deterministically assembled into fully decoded 16-fps H.264, submitted once to Gemini and TwelveLabs, and later carried through genuine browser presentation; provider disagreement and every timing remain non-gating."
anchors: ["bench/cf_video_artifact.py#assemble_cf1_video_artifact", "bench/cf_video_artifact.py#validate_cf_video_artifact", "bench/development_video_judge.py#preflight_development_video_understanding", "bench/development_video_judge.py#run_development_video_understanding"]
related: ["[[Research]]", "[[State]]", "[[CF1-Latent-Pull-and-Smoke]]", "[[Gemini-3.1-Video-Judge-Calibration]]", "[[Pegasus-1.5-Video-Judge-Calibration]]", "[[session-13-development-video-artifact]]"]
---

# CF++1 development video artifact and understanding boundary

## Bottom line

The first complete H100 smoke passed the deterministic artifact and provider
boundary. `assemble_cf1_video_artifact` accepted the exact complete reusable
21-block smoke, double-encoded its 81 PNGs into one 5.0625-second 832×480 H.264
MP4 at 16 fps, fully decoded it, and published a hash-bound development
artifact. `run_development_video_understanding` then submitted the same bytes
once each to Gemini and TwelveLabs.

Video SHA-256 is
`8c892d63cbb4cec9ec7856a352630644c054e27159d092b7642291dc1a4f48fb`;
artifact-manifest SHA-256 is
`1b33d98978a413ef07817d82823bff81be91f6f44c56ac9e9f4d989d76359284`;
and the completed dual-provider manifest is
`e5f48cdfeb3412432f9a80844a04c8d2043498d96ccb8a0f8ad288ed7b3102a0`.
Both outputs say `development-video-understanding-not-gate-evidence` and set
`authorizes_quality_claim:false` and `authorizes_performance_claim:false`.

## Exact artifact contract

The assembler requires a complete `cf1-cuda-smoke` manifest with 21 blocks,
chunk counts `[1] + [4] * 20`, 81 SHA-256-bound nonsymlink PNGs, 832×480 RGB,
and a 16-fps contract. It rejects missing, extra, symlinked, malformed, or
changed frames before publication and rehashes all frames after encoding.

FFmpeg runs twice with the same single-threaded libx264 command: CRF 18,
`yuv420p`, CFR 16 fps, no audio or metadata, fixed track timescale, and bitexact
flags. The two MP4s must be byte-identical under the exact recorded FFmpeg
binary and version. FFprobe must report one total MP4 video stream, no audio,
H.264/yuv420p, 81 decoded frames, and 5.0625 seconds with no error output. A
separate `ffmpeg -xerror` pass must fully decode all 81 frames with empty
stdout/stderr.

Publication uses the platform's atomic no-replace directory rename
(`renameatx_np(..., RENAME_EXCL)` on macOS or
`renameat2(..., RENAME_NOREPLACE)` on Linux) through one same-parent directory
descriptor. File, stage-directory, and parent-directory synchronization make
the publication crash-durable. Revalidation requires the recorded FFmpeg and
FFprobe identities and probes/decodes a private copy of the exact bytes it
returns, closing the path-reopen race.

The published directory contains exactly `manifest.json` and `video.mp4`.
Understanding output must be a sibling or elsewhere; putting it inside the
artifact directory is rejected because it would invalidate that exact set.

## Paid-call and persistence contract

`scripts/cf-video-understand preflight` revalidates the artifact, requires the
exact generation-prompt hash, reads the rubric, and builds and size-checks both
full requests without loading credentials, creating provider attempts, or
sending media. Gemini uses explicit `videoMetadata.fps = 16`. The TwelveLabs
adapter retains a conservative 30 MB encoded limit even though the current
official inline-base64 limit is 36 MB; Pegasus 1.5's documented four-second
minimum is satisfied by the 5.0625-second artifact.

`upload` is the explicit paid boundary. One output-scoped thread/process lock
covers cache validation, request execution, and durable state replacement.
Before transport, the provider state is file-fsynced, atomically renamed to
`in_flight`, and its directory is fsynced. A timeout, parse/version failure,
interrupt, or crash becomes `uncertain` or leaves the durable `in_flight`
record. Normal resume refuses either; only
`--retry-uncertain google|twelvelabs` permits another call. A completed provider
is always re-parsed and reused, so a second-provider failure cannot repeat the
first provider.

Persisted request evidence is rebuilt from exact allowlisted shapes. The
primary base64 must decode to the validated media bytes, extra or future fields
are rejected, and neither encoded media, video bytes, keys, nor authorization
headers enter state. Transport identity is explicit: injected tests record
`injected-unverified`, while the CLI records the built-in urllib adapter. The
CLI timeout is finite and positive and is accurately described as a blocking
HTTP-operation timeout, not a hard wall-clock deadline.

## Review and verification

The actual-file consensus parent `ses_080364d85ffeeLYfXzVHb194db` requested
`claude-opus,kimi-k3,grok` and timed out. SQLite recovery found one concrete
`grok-4.5` final in child `ses_08032f61bffeUT6qgFYrDosEmP`; Fable-5 child
`ses_08032f61effe0FspwNtu9pGT9X` and Kimi-K3 child
`ses_08032f61cffepQzpm7KEss9muu` produced zero output. No Fable rate limit was
recorded, so automatic Opus-4.8 fallback did not apply. Grok's prompt was
truncated before the complete paid path, but its base64/artifact mismatch and
tool-identity findings reproduced and were fixed. Review cost was **$0.146764**.

Independent complete-file audits additionally reproduced duplicate concurrent
calls, missing directory durability, artifact self-invalidation, subtractive
scrubbing, decoder-error acceptance, mutable-path probing, and replace-on-race
publication. Each received a red regression before its fix. Final independent
verdicts are GO with no remaining P0/P1 finding for either the assembler or the
controlled development upload.

Current verification is **13/13** artifact tests, **16/16** development-upload
tests, **332/332** dependency-complete Python discovery tests, **4/4** Node
client tests, green `compileall`, and working CLI help. The artifact suite
includes real libx264 double encoding, full decode, revalidation, and
no-clobber publication on the local FFmpeg 7.1.1 surface.

## First real result and next use

The planned sequence completed exactly: frozen authorizer, one-block smoke,
full 21-block smoke, deterministic assembly, credential-free preflight, then
one explicit upload of the exact MP4 to each provider. Gemini reported
unnatural motion, morphing, and sliding; Pegasus reported smooth natural motion
and stable anatomy on the same bytes. The disagreement is evidence that
Pegasus remains calibration-failed, not a reason to average the ratings or call
the video quality-qualified.

The same generator/decoder subsequently passed two-job persistent-worker
acceptance and one real CF++1 browser presentation run at 81 frames, 21 chunks,
21 post-paint ACKs, visible canvas, server completion, and zero console errors.
Neither provider result nor the MP4/full-smoke/browser completion timing
authorizes quality or performance claims.
