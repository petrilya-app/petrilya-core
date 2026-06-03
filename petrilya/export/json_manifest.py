"""Run manifest JSON for reproducibility.

Captures every parameter that affected a run so the result can be
reproduced or audited later (important for scientific publication
and GMP-relevant workflows).
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from petrilya import __version__


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def build_manifest(
    *,
    image_path: Path,
    masks_shape: tuple[int, int],
    n_objects: int,
    elapsed_seconds: float,
    engine_name: str,
    engine_params: dict,
    scale_um_per_px: float | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "petrilya_version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
        },
        "input": {
            "path": str(image_path),
            "filename": image_path.name,
            "sha256": file_sha256(image_path),
            "size_bytes": image_path.stat().st_size,
        },
        "engine": {
            "name": engine_name,
            "params": engine_params,
        },
        "result": {
            "n_objects": n_objects,
            "masks_shape": list(masks_shape),
            "elapsed_seconds": round(elapsed_seconds, 4),
            "scale_um_per_px": scale_um_per_px,
        },
    }


def write_manifest(manifest: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
