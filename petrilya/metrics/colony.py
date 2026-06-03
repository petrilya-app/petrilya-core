"""Compute per-colony metrics from segmentation masks."""

from __future__ import annotations

import numpy as np
from skimage import measure


def compute_colony_metrics(masks: np.ndarray) -> list[dict]:
    """Compute area, centroid, equivalent diameter for each colony."""
    if masks.max() == 0:
        return []
    props = measure.regionprops(masks)
    return [
        {
            "id": int(p.label),
            "area_px": int(p.area),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
            "equivalent_diameter_px": float(p.equivalent_diameter),
            "eccentricity": float(p.eccentricity),
            "solidity": float(p.solidity),
        }
        for p in props
    ]
