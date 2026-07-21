---
type: research
status: active
date: 2026-07-20
description: "The minimal CF++1 plus rolling-TAEHV source/model stack is byte-complete and fail-closed; its frozen runtime, safe direct loaders, actual FA2/Torch binding, real 21-block CUDA smoke, persistent-worker lifecycle, and real-browser presentation have now executed on one H100."
anchors: ["bench/model_asset_preflight.py#verify_model_assets", "bench/cf_cuda_adapter.py#build_cf1_runtime", "bench/cf_cuda_adapter.py#validate_cf1_runtime_provenance", "bench/cf_cuda_adapter.py#_build_checkpoint_only_generator", "bench/cf_cuda_adapter.py#_build_pinned_text_encoder", "bench/cf_cuda_adapter.py#_build_pinned_taehv"]
related: ["[[Research]]", "[[State]]", "[[CF1-H100-Runtime-Preflight]]", "[[CF1-Rolling-TAEHV-Session]]", "[[Streaming-Service-Boundary]]", "[[Gotcha-Rolling-TAEHV-Context-Trim]]", "[[session-9-pinned-assets-and-cuda-bootstrap]]", "[[session-10-rolling-cuda-session-core]]", "[[session-11-runtime-preflight]]"]
---

# Pinned CF++1 CUDA bootstrap

## Bottom line

The minimal stack for the historical CF++1 plus corrected rolling-TAEHV path is
complete and byte-verified. `scripts/model-asset-preflight`
returns `ready: true`, the exact upstream checkout is clean, and its filesystem
inventory contains no unregistered path. This closes the missing-source and
missing-weight blocker.

`bench/cf_cuda_adapter.py` defers third-party imports until after the asset
gate, directly
constructs the pinned causal Wan architecture, strictly assigns the CF++ EMA
state, constructs the text encoder and TAEHV through explicit safe loaders,
injects a latent passthrough into `CausalInferencePipeline`, and emits verified
whole-stack provenance. `bench/cf_runtime_preflight.py` sits in front with a
frozen schema-v2 runtime/evidence lock, tokenizer sentinel, and capacity gate;
`bench/cf_cuda_session.py` consumes that provenance in the rolling decoder.
The unchanged boundary has now loaded the real tensors and completed the exact
45-forward, 21-block/81-PNG smoke plus two persistent-worker jobs on one H100.
Those are development execution and lifecycle proofs, not quality, performance,
or browser-presentation authorization.

## Immutable inventory

| Asset | Immutable source pin | Bytes | SHA-256 |
|---|---|---:|---|
| `configs/default_config.yaml` | `thu-ml/Causal-Forcing@8db419e341e5fc52542c0b2c4542728420ddfb4a` | 403 | `18303f1bacbfbebaa02893cc8e106941753f21a7d1a7109519a90762375ee772` |
| `configs/causal_forcing_dmd_framewise_1step.yaml` | `thu-ml/Causal-Forcing@8db419e341e5fc52542c0b2c4542728420ddfb4a` | 1,791 | `0b1e43656ba4b0544e85b047be46a79b156d683bfb06eaf99c8817826967623d` |
| `demo_utils/taehv.py` | `thu-ml/Causal-Forcing@8db419e341e5fc52542c0b2c4542728420ddfb4a` | 14,157 | `865b1dd0516682c02352bd76bce2ad0472588330ea9fab30ca0895630f263244` |
| `checkpoints/taew2_1.pth` | `madebyollin/taehv@1a88a7dcafa06f46866661c0d687654aafa5521b` | 22,678,901 | `d26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e` |
| `checkpoints/causal-forcing++/framewise-1step.pt` | `zhuhz22/Causal-Forcing@a6a8f0e3bbdea1044fc6fef09c9cb9f648bf1bc3` | 5,676,220,819 | `bdb1b475fc88d528f510158a0990cd457f02c661b527ceb11ca9e4728533e2d0` |
| `wan_models/Wan2.1-T2V-1.3B/config.json` | `Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a` | 249 | `ab37994c43740513f94b3ba6233a784035a67b43c8cde83c8f31aa90468c67ce` |
| `wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth` | `Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a` | 11,361,920,418 | `7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d` |
| `google/umt5-xxl/special_tokens_map.json` | `Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a` | 6,623 | `7b8a9f5040adb67b5805abdfd42c1f8d0f3d0e711f10726580eb3789cd0ad61d` |
| `google/umt5-xxl/spiece.model` | `Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a` | 4,548,313 | `e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458` |
| `google/umt5-xxl/tokenizer.json` | `Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a` | 16,837,417 | `6e197b4d3dbd71da14b4eb255f4fa91c9c1f2068b20a2de2472967ca3d22602b` |
| `google/umt5-xxl/tokenizer_config.json` | `Wan-AI/Wan2.1-T2V-1.3B@37ec512624d61f7aa208f7ea8140a131f93afc9a` | 61,728 | `ed9a3a8b0faa71a70a32847e0435fe036e6e112d4df4edb7bb48a921e344dc05` |

The source checkout itself is exactly
`thu-ml/Causal-Forcing@8db419e341e5fc52542c0b2c4542728420ddfb4a`.

The lock is `bench/model_assets/cf1-rolling-taehv-v1.lock.json`, currently
SHA-256 `0aee8671f8e3b30286b689a16f6f4a355f917772c16e599cec75a49e89057967`.
The stock Wan diffusion checkpoint (5,676,070,424 bytes) and stock Wan VAE
(507,609,880 bytes) were deliberately not downloaded because the bootstrap does
not execute either upstream constructor that opens them.

## Fail-closed asset gate

The lock parser rejects duplicate or unknown fields, unsafe paths, duplicate
IDs/paths, booleans disguised as integers, NULs, malformed commits, and
malformed hashes. The verifier:

- scrubs inherited Git control variables and checks the exact HEAD;
- inventories every checkout path rather than trusting `.gitignore`, allowing
  only tracked source plus the exact lock-listed assets;
- opens every path component with `O_NOFOLLOW`, requires a regular file, hashes
  through the same descriptor, and compares pre/post/path-entry fingerprints;
- reports missing, wrong-size, wrong-hash, non-regular, unreadable, and
  changed-during-verification states separately; and
- returns ready only when the source and every asset verify.

This blocks ignored bytecode injection and unregistered files placed beside
ignored weights. The bootstrap repeats the complete gate after model loading to
catch ordinary replacement or mutation during startup. It is detection, not a
defense against an adversarial local process that can rewrite and swap files
between verification and later path-based imports/loads. The current threat
model is a trusted local worker with no concurrent writer; revisit this boundary
before accepting remote, untrusted, or multi-tenant model bytes.

The persistent-process seam was hardened in parallel: it rejects false
isolation and nonempty sensitive-environment claims, loads a single shared
protocol version and 21-latent cap under `python -I`, equality-fences that cap in
HELLO, and binds the worker entrypoint plus shared protocol-definition bytes in
its length-prefixed bundle digest. Those checks strengthen a trusted local
worker boundary; they do not turn it into an OS sandbox.

## Bootstrap contract

- `CausalWanModel` is built from the byte-pinned Wan `config.json` with the
  historical global-attention values `local_attn_size=-1`, `sink_size=0`.
- A named wrapper bypasses `WanDiffusionWrapper.__init__`, so
  `CausalWanModel.from_pretrained` and the stock diffusion checkpoint are not
  touched. The CF++ archive is loaded with `weights_only=True`, `mmap=True`,
  normalized only for the observed FSDP prefix, checked against the model's
  complete key set, and installed with `strict=True, assign=True`.
- A pinned text wrapper bypasses the upstream `weights_only=False` loader. It
  constructs UMT5, loads the exact state with the same safe mmap/strict/assign
  posture, and points the tokenizer at the four independently pinned files.
- `CausalInferencePipeline` receives generator, text encoder, and latent
  passthrough objects, so its stock Wan VAE constructor is not called. TAEHV is
  instantiated with `checkpoint_path=None`, then the exact registered state is
  loaded explicitly with `weights_only=True`, `mmap=True`, complete key-set
  equality, `strict=True`, and `assign=True`.
- The actual two-file OmegaConf merge was exercised locally with OmegaConf
  2.3.0. Required runtime fields validate as CF++1: one normal denoising step,
  four first-chunk steps, warped schedule enabled, one latent per block, global
  attention, and `timestep_shift=5.0`. Canonical JSON of that observed merge is
  SHA-256 `54a3f8975721fabac17edcd022fd96a763e371021c9fe30452e5ef890d3a5b06`.
  That digest is now an exact runtime pin rather than a candidate.
- Bootstrap provenance binds the exact source commit, asset-lock digest,
  effective-config digest, and post-load observed identity for all 11 assets.
  Its recomputed identity also includes a canonical guard-manifest digest over
  the runtime preflight, adapter, rolling session, generation preflight,
  model-asset preflight, and streaming-service modules. The session rejects a
  runtime if any field or recomputation differs. Provenance additionally binds
  the tokenizer sentinel and actual selected attention backend; the decoder
  requires the exact Torch module object used by the verified runtime.

Static pickle inspection of the exact generator archive found only
`OrderedDict`, `_rebuild_tensor_v2`, and `FloatStorage` globals, with 825 FSDP
generator keys. The exact UMT5 archive likewise contains only `OrderedDict`,
`_rebuild_tensor_v2`, and `BFloat16Storage`. That supported the safe loader
choice; the later frozen H100 smoke and persistent-worker acceptance supplied
the previously missing real PyTorch load and runtime-compatibility proof.

## Review history and current boundary

The first panel parent `ses_0812d3743ffeNqf9A62rBV21dO` timed out. Its Fable
child `ses_0812a1638ffe4F1u90mEwNuRj1` ended in a malformed recap without a
verdict, Kimi produced no usable result, and Grok child
`ses_0812a1633ffeOG052QDWsU6o8w` returned GO-WITH-FIXES. Its in-scope
symlink/TOCTOU and latent-cap findings were closed; process isolation was
already enforced and was made executable in tests.

Post-fix consensus parent `ses_0811ac99dffeQeMEbUc0f1dk0N` also timed out, but
direct session recovery retained a concrete Fable-5 GO-WITH-FIXES final from
`ses_08116fde9ffeeLTdM3BXbn8FM7` and a concrete Grok-4.5 GO-WITH-FIXES final
from `ses_08116fde7ffejKAMoyuca0BwSI`; Kimi-K3 session
`ses_08116fde8ffeg5L527sIUn1XC1` produced no output. Their ignored-path,
inherited-Git-environment, NUL-path, descriptor-close, and protocol-bundle
findings reproduced and were closed. The residual process-image attack requires
a malicious actor able to rewrite trusted local worker code and lie in HELLO;
it is outside the explicitly documented trusted-worker threat model, not an OS
sandbox guarantee.

A separate local Claude CLI review using `--model opus` rated the bootstrap
GO-WITH-FIXES and confirmed that the stock diffusion and VAE loads are genuinely
bypassed. Its actionable config, timestep-shift, tokenizer-inventory, and safe
load findings were incorporated. At that review stage, mmap/assign did not
settle whether fully initialized FP32/CPU module residency would fit, so real
boot still required measured admission or meta/empty initialization. Runtime
dependency pins and tokenizer sentinel IDs also remained open then. The CLI did not expose a concrete
model-version identifier.

[[CF1-H100-Runtime-Preflight]] now records the completed freeze. Runtime
evidence SHA is `8209043b…`, frozen-lock SHA is `d4d163d6…`, and the locked
runtime/native/bound-probe identities are `858c1600…`, `8884c4be…`, and
`070efce1…`. The unchanged authorizer passed before construction. The initial
real execution also proved the existing mmap/assign load strategy fit the
admitted host/HBM envelope; meta/empty initialization was not required for this
bounded path.

The rolling decoder review is recorded in [[CF1-Rolling-TAEHV-Session]]. Parent
`ses_080ea9c12ffejex9NyF4Yi7sk0` timed out without synthesis; direct recovery
found concrete Fable-5 and Grok-4.5 GO-WITH-FIXES finals, while Kimi-K3 stalled
with zero tokens. Fable succeeded, so no Opus fallback occurred. A local Claude
CLI Opus review then mutation-tested the session and every reproducible
mocked-core gap was closed. This remains partial/fallback historical review,
not completed three-family consensus.

The complete smoke manifest SHA is `d363fbe1…`; it binds 21 chunks and 81
decodable 832×480 PNGs. Later guard hardening changed the recomputed stack from
the frozen one-shot identity `3fea7253…` to current `349c79f4…` without changing
the frozen model/runtime bytes. The worker correctly rejected the stale
expected identity; after the caller adopted the independently derived current
identity, manifest `30bfe91b…` proved two distinct-seed jobs on the same
PID/instance and forced-death poison/reap.

The separate opt-in real backend has now crossed the browser boundary without
changing the fake default: the genuine CF++1 H100/frozen-runtime page completed
81 visible frames in 21 chunks with 21 post-paint ACKs, server completion, a
visible canvas, and zero console errors. That result validates the bounded
execution-to-presentation chain, not quality or throughput.
