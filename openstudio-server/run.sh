#!/usr/bin/env bash
# openstudio-server bootstrap + launch, for the RunPod pytorch image
# (torch 2.8.0+cu128, python 3.12). Idempotent: safe to re-run after pod restart.
#
#   pod$    ./run.sh                     # install (first run only) + serve on 127.0.0.1:8765
#   pod$    ./run.sh --selfcheck         # server-side smoke, no client needed
#   laptop$ ssh -N -L 8765:127.0.0.1:8765 -p <ssh-port> root@<pod-ip> -i ~/.ssh/id_ed25519
#           # (must be the pod's DIRECT TCP ssh; the ssh.runpod.io proxy cannot -L forward)
#
# Extra args pass through to server.py (e.g. ./run.sh --prompt "..." --port 8765).
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
# Everything mutable lives on /workspace (the container layer resets on pod
# stop): HF_HOME so the ~2.5 GB of weights download once, and a venv so the
# pip installs survive restarts too. The venv also sidesteps PEP 668 — the
# ubuntu2404 image marks system python externally-managed, so a bare
# `pip install` refuses. --system-site-packages keeps the image's torch stack
# visible while our pins install into the venv only; the postflight below
# still proves torch came through untouched.
if [[ -d /workspace ]]; then
  export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
  VENV="${VENV:-/workspace/openstudio-venv}"
  if [[ ! -x "${VENV}/bin/python" ]]; then
    "$PYTHON" -m venv --system-site-packages "${VENV}"
  fi
  PYTHON="${VENV}/bin/python"
fi

# ---- preflight: the image's torch stack must already be importable ----------
torch_before="$("$PYTHON" - <<'PY'
import torch, torchvision
print(f"{torch.__version__}|{torchvision.__version__}")
PY
)" || { echo "FATAL: torch/torchvision not importable — run on the RunPod pytorch image (or set PYTHON)." >&2; exit 1; }
echo "preflight: torch|torchvision = ${torch_before}"

# ---- install (skipped when already satisfied; pip is fast on no-ops) --------
"$PYTHON" -m pip install --no-cache-dir -r requirements.txt
# streamdiffusion pins diffusers==0.24.0 (+fire/omegaconf/onnx…) in setup.py.
# We already installed its real runtime deps from requirements.txt, so take the
# package dep-free — pip must never get a chance to touch torch. The PyPI
# release (0.1.1) matches git main; git is used as the canonical source.
"$PYTHON" -m pip install --no-cache-dir --no-deps \
  "streamdiffusion @ git+https://github.com/cumulo-autumn/StreamDiffusion.git@main"

# ---- postflight: prove pip did not shadow the system torch ------------------
torch_after="$("$PYTHON" - <<'PY'
import torch, torchvision
print(f"{torch.__version__}|{torchvision.__version__}")
PY
)"
if [[ "${torch_after}" != "${torch_before}" ]]; then
  echo "FATAL: pip replaced the torch stack (${torch_before} -> ${torch_after})." >&2
  echo "       Reinstall the image's torch build before serving — do NOT serve on the wrong wheel." >&2
  exit 1
fi
"$PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() is False — wrong pod / driver problem"
import cv2, websockets, diffusers, transformers, huggingface_hub
from streamdiffusion import StreamDiffusion
from streamdiffusion.image_utils import postprocess_image
print(f"postflight OK: cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)} "
      f"diffusers={diffusers.__version__} hf_hub={huggingface_hub.__version__}")
PY

exec "$PYTHON" server.py "$@"
