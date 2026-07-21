# openstudio-server

The open-model counterpart to studio's Lucy 2.5 mode. The browser sends camera frames as JPEG over one WebSocket, this server restyles each frame with StreamDiffusion (`stabilityai/sd-turbo`, 512×512, 2-step img2img, TAESD tiny VAE, fp16) on a rented GPU, and streams restyled JPEG frames back. The prompt hot-swaps mid-session without reconnecting (~5–20 ms, one CLIP forward).

Honest v1 tradeoff: per-frame img2img has no temporal modeling, so output flickers more than Lucy. In exchange it is the only open config with a proven single-4090 webcam-realtime record, and the wire contract is engine-agnostic — `hello` advertises `pipeline`/`model`, so a temporally-coherent engine (StreamDiffusionV2 wan_causal_dmd_v2v) can replace it server-side without touching the client.

## The pod

One RunPod **RTX 4090** (secure cloud, $0.69/hr) on the stock **`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`** image (torch 2.8.0+cu128, Python 3.12). Two hard requirements at rental time:

- **Direct-TCP SSH** — the pod needs a public IP with a mapped TCP port for 22 (`ports: ["22/tcp"]`). The `ssh.runpod.io` proxy cannot `-L` forward, and the tunnel below is the only way in.
- **A `/workspace` volume** — `run.sh` puts `HF_HOME` there so the ~2.5 GB of model weights survive pod stop/start.

Ship the code and run the GPU smoke before ever involving a browser:

```bash
rsync -rlpt --no-owner --no-group --exclude .venv-test \
  -e "ssh -i ~/.ssh/albertdbio.pem -p <ssh-port>" \
  ./ root@<pod-ip>:/workspace/openstudio-server/

ssh -i ~/.ssh/albertdbio.pem -p <ssh-port> root@<pod-ip> \
  'cd /workspace/openstudio-server && ./run.sh --selfcheck'
```

`run.sh` is idempotent: it verifies the image's torch before AND after installing `requirements.txt` (fails loud if any wheel shadowed torch 2.8.0+cu128 — the reason streamdiffusion is installed `--no-deps`), downloads sd-turbo + TAESD into the workspace HF cache, then `--selfcheck` pushes synthetic 720p frames through the exact serve path with a mid-run prompt swap and prints `infer_ms` p50/p95 + `est_fps`. `scp` the written in/out JPEGs back and look at them — success is `est_fps ≥ 10` **and** visibly restyled, non-collapsed output that changes at the prompt swap.

## Serve and tunnel

```bash
# on the pod — binds 127.0.0.1:8765 ONLY, never a routable interface
cd /workspace/openstudio-server && ./run.sh

# on the laptop — the key-authenticated tunnel IS the auth boundary
ssh -N -L 8765:127.0.0.1:8765 -p <ssh-port> root@<pod-ip> -i ~/.ssh/albertdbio.pem
```

Smoke the live link from the laptop before the browser: with the tunnel up, `CONTRACT_TEST_PORT=8765 uv run --with opencv-python-headless --with numpy --with websockets python contract_test.py` drives hello/frame/prompt/stats over the real GPU path (its own local server fails to bind the taken port and the checks run against the tunnel).

## How the studio connects

The studio's open mode opens `ws://localhost:8765` (user-editable) with `binaryType = "arraybuffer"`, verifies `hello.proto === 1`, sizes its canvases from `hello.width`/`hello.height`, sends the UI's current prompt, then ticks frames at ~15 fps — capture-at-send, time-based pacing, never ack-based. Binary frames carry a little-endian header (in: `u8 magic 0x01, u32 seq, f64 capture_ts_ms`; out adds `f32 infer_ms`); the echoed timestamp gives glass-to-glass latency off one clock. Both sides are newest-frame-wins: the server keeps a 1-slot mailbox in and at most one pending frame out, so nothing ever queues behind a slow link.

The authoritative wire contract lives in `server.py`'s module docstring; `contract_test.py` is its executable form (15 checks against a fake-pipeline boot) and `test_server_protocol.py` is the white-box half (in-process handler + instrumented stub pipeline). Keep both green on every `server.py` change.

## Local dev without a GPU

```bash
# transport-only server on the laptop (no torch, no model)
python server.py --pipeline fake

# black-box contract suite (boots its own fake-pipeline server)
uv run --with opencv-python-headless --with numpy --with websockets python contract_test.py

# white-box protocol suite
uv venv .venv-test --python 3.12
uv pip install -p .venv-test/bin/python pytest pytest-asyncio websockets==15.0.1 numpy opencv-python-headless
.venv-test/bin/python -m pytest test_server_protocol.py -q
```

## Knobs and tuning

Runtime: `{"type":"config","jpeg_quality":30–95}` is v1's only live knob. Everything else is a **boot flag** — StreamDiffusion sizes its latent buffers in `prepare()`, so model/resolution/t_index/cfg cannot change on a live pipeline. `hello` reports every lever, so the client (and you, via the browser console) can always see exactly what config is serving. If fps lands under 10 at 512×512 (unexpected on a 4090): `--t-index 45` gives 1-step, then lower `--jpeg-quality`; TensorRT stays deferred past v1.

### Quality lever cheat-sheet

Defaults reproduce the original v1 behavior exactly (sd-turbo, `[35,45]`, TAESD, no cfg, add-noise, no filter). Each lever, what it buys, what it costs:

| Flag | Effect | Cost |
|---|---|---|
| `--t-index i0,i1[,i2]` | Restyle strength ≈ (50−i0)/50: `35,45`=0.30 subtle (default), `32,45`=0.36, `28,42`=0.44, `25,40`=0.50 strong. More strength = more restyle punch *and* more per-frame variance. | Free for len 2; len 3 (`25,35,45`) +50% UNet (~8 fps). Single `45` halves UNet. |
| `--cfg-type self --guidance-scale 1.4 [--delta 1.0]` | RCFG "self": virtual negative guidance from the stock-noise residual — stronger prompt adherence. **`--guidance-scale` must be > 1.0 or it's a no-op** (server warns); sweep 1.2–1.6, delta 0.5–1.0. Prompt hot-swap stays safe (cond-only embeds). `initialize`/`full` are rejected — they break `update_prompt`. | ~0 (elementwise math, no extra UNet pass) |
| `--model <id> [--lcm-lora]` | SD1.5-family checkpoint swap. Finetunes (`Lykon/dreamshaper-8`, kohaku-v2.1 — the community quality recipe) **need `--lcm-lora`** (fuses `latent-consistency/lcm-lora-sdv1-5`); `*turbo*` models must not use it. | ~0 fps (same UNet); one-time weight download |
| `--noise-mode deterministic` | `do_add_noise=False` — the upstream authors' own vid2vid anti-flicker latent path. Stabler, slightly softer/less textured. | 0 |
| `--similar-filter 0.98 [--similar-max-skip 10]` | Skip inference when the input barely changed and replay the previous output — kills sitting-still shimmer outright. Slight pop when motion resumes. | Negative (saves GPU on static scenes) |
| `--full-vae` | Full SD VAE instead of TAESD: crisper detail, truer color (TAESD is the known softness/color-shift source, worst on faces). | +18–25 ms/frame → ~9–10 fps |
| `--seed N` | Init-noise seed. The noise tensor is sampled once and reused every frame (that fixedness is itself the anti-flicker default). | 0 |

Still true under every lever combo: negative prompts are a no-op (`none` and `self` both build cond-only embeds — no uncond branch), and resolution is the worst quality-per-ms lever (models are 512-trained; 640² already eats the whole fps margin).

## Quality bench — `bench_quality.py`

Offline sweep that measures each lever combo before you bet a live session on it. It runs every config in the `CONFIGS` list (seeded from the scout ranking: baseline, RCFG, dreamshaper+LCM-LoRA, temporal bundle, strength probe, full-VAE) over simulated webcam sequences — hold / slow pan+zoom / hold with per-frame sensor noise, over 4 photo-like scenes (CC0 photos downloaded once, synthetic photo-like fallbacks with `--no-download` or on any fetch failure) — and scores each config on:

- **fps / infer_ms p50/p95** — wall time around `pipeline.process`, same span the server reports
- **flicker** — mean |out_t − out_{t−1}| restricted to pixels whose *input* pair was static (output churn not explained by input motion), split hold-vs-motion, plus `dup_ratio` (bit-identical outputs = similar-filter freezes)
- **clip / Δclip** — CLIP prompt adherence of outputs, and gain over the raw inputs (`open_clip` if present, else the transformers CLIP already in the deps — first use downloads ~600 MB into the HF cache; null if neither imports, ranking then falls back to flicker-only)

```bash
# pod (same venv run.sh bootstraps; HF_HOME so weights come from/land in the workspace cache)
cd /workspace/openstudio-server
HF_HOME=/workspace/hf-cache /workspace/openstudio-venv/bin/python bench_quality.py

HF_HOME=/workspace/hf-cache /workspace/openstudio-venv/bin/python bench_quality.py \
  --configs baseline,rcfg-self          # subset; --list shows all + their server flags

# laptop (no GPU/torch): proves the harness end-to-end against the fake pipeline
python bench_quality.py --pipeline fake --no-download --no-clip --frames 12
```

It writes `bench_out/`: `results.md` (ranked table — configs under `--fps-floor` 8 sink below all that meet it), `results.json`, per-config `sheet_<name>.jpg` contact sheets (input rows vs restyled rows per scene × prompt), and `compare.jpg` (one settled frame per config, ranked). Every row carries a `server_flags` string — boot the production server in exactly the benched config:

```bash
./run.sh --t-index 32,45 --cfg-type self --guidance-scale 1.4 --noise-mode deterministic --similar-filter 0.98
```

The scores are cheap proxies, not verdicts: eyeball the winner's contact sheet (especially `dup_ratio` — a high value means the similar-filter froze the output, which zeroes flicker by construction), then A/B it live. First full sweep also downloads dreamshaper-8 + LCM-LoRA (~2 GB); expect ~15–30 min on the 4090 including loads.

## Security and cost

There is no in-protocol auth: the server must only ever bind `127.0.0.1`, and the SSH tunnel is the sole route in. Add real auth before binding anything else. The pod bills $0.69/hr while running — stop it when idle; the HF cache on `/workspace` survives and `run.sh` re-bootstraps in seconds.
