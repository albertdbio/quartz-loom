"""Standard-library-only definitions shared by the process supervisor and worker."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

WORKER_PROTOCOL_VERSION = "realtime-video.process-worker.v1"
WORKER_PROTOCOL_MAX_LATENT_FRAMES = 21


def worker_bundle_sha256(
    worker_script: Path,
    companion_paths: Sequence[Path] = (),
) -> str:
    """Bind the entrypoint, protocol, and any executable companion modules."""

    digest = hashlib.sha256()
    paths = [
        (b"protocol", Path(__file__)),
        (b"worker", Path(worker_script)),
    ]
    seen_labels = {label for label, _path in paths}
    for companion in companion_paths:
        path = Path(companion)
        try:
            label = b"companion:" + path.name.encode("utf-8")
        except UnicodeError as error:
            raise ValueError("worker companion name is not valid UTF-8") from error
        if label in seen_labels:
            raise ValueError("worker bundle contains a duplicate companion name")
        seen_labels.add(label)
        paths.append((label, path))
    for label, path in paths:
        payload = path.read_bytes()
        digest.update(len(label).to_bytes(4, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
