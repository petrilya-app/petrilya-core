"""Compute per-colony metrics from segmentation masks."""

from __future__ import annotations

import numpy as np
from skimage import measure


def compute_colony_metrics(
    masks: np.ndarray,
    scale_um_per_px: float | None = None,
) -> list[dict]:
    """Compute area, centroid, equivalent diameter for each colony.

    If ``scale_um_per_px`` is provided, also report area_um2 and
    equivalent_diameter_um in physical units.
    """
    if masks.max() == 0:
        return []
    props = measure.regionprops(masks)
    out: list[dict] = []
    for p in props:
        rec = {
            "id": int(p.label),
            "area_px": int(p.area),
            "centroid_y": float(p.centroid[0]),
            "centroid_x": float(p.centroid[1]),
            "equivalent_diameter_px": float(p.equivalent_diameter_area),
            "eccentricity": float(p.eccentricity),
            "solidity": float(p.solidity),
        }
        if scale_um_per_px is not None and scale_um_per_px > 0:
            rec["area_um2"] = float(p.area) * scale_um_per_px**2
            rec["equivalent_diameter_um"] = (
                float(p.equivalent_diameter_area) * scale_um_per_px
            )
        out.append(rec)
    return out
