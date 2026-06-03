"""Cellpose-based segmentation engine for colony counting."""

from __future__ import annotations

import numpy as np
from cellpose import models


class CellposeEngine:
    """Wrapper around Cellpose for colony segmentation."""

    def __init__(self, model_type: str = "cyto3", use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu
        self.model_type = model_type
        self.model = models.Cellpose(gpu=use_gpu, model_type=model_type)

    def segment(
        self,
        image: np.ndarray,
        diameter: float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run segmentation. Returns (masks, estimated_diameter)."""
        masks, _flows, _styles, diams = self.model.eval(
            image,
            diameter=diameter,
            channels=[0, 0],
        )
        return masks, float(diams) if np.isscalar(diams) else float(diams[0])
