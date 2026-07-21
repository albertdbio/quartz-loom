---
type: handoff
status: active
session: 25
date: 2026-07-21
description: "Brought the openstudio-server sd-turbo StreamDiffusion engine live on a rented 4090 pod and verified selfcheck, tunnel contract, and real-imagery restyle end-to-end; pod intentionally left running for client integration."
branch: main
key_commits: []
prior_handoff: "session-24-sgmd-degraded-displacement"
---

# Session 25 Handoff — openstudio 4090 open-model server live

- Deployed `openstudio-server/` to pod `52hu4efmx6bym0` (`openstudio-4090`, US-NC-1, $0.69/hr, stock torch-2.8.0+cu128 image). `run.sh` grew a `--system-site-packages` venv at `/workspace/openstudio-venv` because the ubuntu2404 image enforces PEP 668 on system pip; the venv also survives pod stops, and the torch pre/post guards proved the image stack was never shadowed.
- GPU selfcheck PASS (infer_ms p50 70.27 / p95 102.25, est_fps 10.5 — above the ≥10 bar, interactive not real-time; prompt swap 10.28 ms) with eyeballed JPEGs: correct cover-fit, non-collapsed prompt-responsive output, and out_000 carrying warmup frame-9 content — pixel proof of the discard-2 warmup rule. Live tunnel smoke: `contract_test.py` 12/12 against the real GPU, real-imagery probe 8/8 frames at e2e p50 152 ms with an unmistakable watercolor restyle.
- New white-box net `test_server_protocol.py` (8/8 in `.venv-test`): in-process handler + instrumented gate-frozen stub makes newest-frame-wins and prompt coalescing deterministic; mutation-checked red on ts-echo and oldest-wins. `contract_test.py` gained `CONTRACT_TEST_SKIP_FIREHOSE=1` for narrow tunnel links (firehose burst starves the test client's own keepalive at ~5 Mbit/s up); local 15/15 unchanged. Pod left RUNNING deliberately (zero free 4090s in US-NC-1 — a stop risks the GPU); tunnel: `ssh -N -L 8765:127.0.0.1:8765 -p 10394 root@103.196.86.87 -i ~/.ssh/albertdbio.pem`. Spend ~$0.8 logged; no commit.
