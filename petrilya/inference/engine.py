"""Cellpose-based segmentation for Petrilya.

Uses Cellpose with the cyto3 model. Weights are auto-downloaded from
cellpose.org on first use and cached in ``~/.cellpose/models/``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a ``uint8`` grayscale view of a 2D or RGB(A) input."""
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[-1] >= 3:
        rgb = image[..., :3].astype(np.float32)
        Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return np.clip(Y, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {image.shape}")


class CellposeEngine:
    """Thin wrapper around ``cellpose.models.Cellpose``.

    Returns ``(labels, estimated_diameter_px)`` from ``segment``.
    """

    engine_name = "cellpose-cyto3"

    def __init__(
        self,
        *,
        model_type: str = "cyto3",
        use_gpu: bool = False,
        **_: Any,
    ) -> None:
        from cellpose import models

        self.model_type = model_type
        self.use_gpu = use_gpu
        self.model = models.Cellpose(gpu=use_gpu, model_type=model_type)

    def segment(
        self,
        image: np.ndarray,
        diameter: float | None = None,
    ) -> tuple[np.ndarray, float]:
        # Cellpose accepts both grayscale and RGB; we normalise to grayscale
        # so the UI's colour display can't accidentally change inference.
        gray = to_grayscale(image)
        masks, _flows, _styles, diams = self.model.eval(
            gray, diameter=diameter, channels=[0, 0]
        )
        masks = masks.astype(np.int32)
        diam = float(diams) if np.isscalar(diams) else float(diams[0])
        return masks, diam

    def describe(self) -> dict[str, Any]:
        return {
            "engine": "cellpose",
            "model": self.model_type,
            "params": {"use_gpu": self.use_gpu},
        }
