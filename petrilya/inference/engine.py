"""Cellpose-based segmentation for Petrilya.

Uses Cellpose with the cyto3 model (weights are auto-downloaded from
cellpose.org on first use and cached in ``~/.cellpose/models/``).

We post-process Cellpose's raw output with a shape filter, because
cyto3 was trained on microscopy of individual cells, NOT on macro
photos of Petri dishes — so on a phone photo of a plate it tends to
ALSO latch onto marker-pen text strokes, rim scratches, and stray
agar artefacts. Real colonies are nearly round (low eccentricity,
high solidity) while those artefacts are elongated or jagged, so a
single eccentricity+solidity gate removes most false positives.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Shape filtering thresholds
#
# Empirically picked from looking at regionprops of typical false
# positives (marker text strokes, dish-rim arcs, hairline cracks) on
# the AGAR sample images. Tightening any of these further loses real
# colonies; relaxing them lets text back in. Exposed as class attrs so
# they're easy to override for special cases without forking the file.
# ---------------------------------------------------------------------

MAX_ECCENTRICITY = 0.85       # 0=circle, 1=line.  Text strokes are >0.9.
MIN_SOLIDITY     = 0.78       # area / convex-hull area. Jagged shapes <0.7.
MIN_AREA_PX      = 30         # speckle floor; smaller is almost certainly noise.
MAX_AREA_FRAC    = 0.25       # any single 'colony' > 25% of plate area is wrong.


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a ``uint8`` grayscale view of a 2D or RGB(A) input."""
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[-1] >= 3:
        rgb = image[..., :3].astype(np.float32)
        Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return np.clip(Y, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def filter_by_shape(
    masks: np.ndarray,
    *,
    max_eccentricity: float = MAX_ECCENTRICITY,
    min_solidity: float = MIN_SOLIDITY,
    min_area_px: int = MIN_AREA_PX,
    max_area_frac: float = MAX_AREA_FRAC,
) -> tuple[np.ndarray, dict[str, int]]:
    """Drop regions that don't look like colonies.

    Returns ``(filtered_labels, stats)`` where stats counts how many
    were removed for each reason — useful for honest UI reporting
    ("before filter: 264, kept: 18").
    """
    from skimage import measure, segmentation

    if masks.max() == 0:
        return masks.astype(np.int32, copy=False), {
            "before": 0, "kept": 0,
            "dropped_eccentricity": 0,
            "dropped_solidity": 0,
            "dropped_area": 0,
        }

    image_area = float(masks.size)
    max_area_px = int(image_area * max_area_frac)

    out = masks.astype(np.int32, copy=True)
    stats = {
        "before": int(masks.max()),
        "dropped_eccentricity": 0,
        "dropped_solidity": 0,
        "dropped_area": 0,
    }
    for prop in measure.regionprops(out):
        drop_reason = None
        if prop.eccentricity > max_eccentricity:
            drop_reason = "dropped_eccentricity"
        elif prop.solidity < min_solidity:
            drop_reason = "dropped_solidity"
        elif prop.area < min_area_px or prop.area > max_area_px:
            drop_reason = "dropped_area"
        if drop_reason:
            out[out == prop.label] = 0
            stats[drop_reason] += 1

    # Compact the IDs so the table shows 1..N with no gaps.
    out = segmentation.relabel_sequential(out)[0].astype(np.int32)
    stats["kept"] = int(out.max())
    return out, stats


class CellposeEngine:
    """Thin wrapper around ``cellpose.models.Cellpose`` + shape post-filter."""

    engine_name = "cellpose-cyto3"

    def __init__(
        self,
        *,
        model_type: str = "cyto3",
        use_gpu: bool = False,
        shape_filter: bool = True,
        **_: Any,
    ) -> None:
        from cellpose import models

        self.model_type = model_type
        self.use_gpu = use_gpu
        self.shape_filter = shape_filter
        self.model = models.Cellpose(gpu=use_gpu, model_type=model_type)
        self.last_filter_stats: dict[str, int] | None = None

    def segment(
        self,
        image: np.ndarray,
        diameter: float | None = None,
    ) -> tuple[np.ndarray, float]:
        gray = to_grayscale(image)
        masks, _flows, _styles, diams = self.model.eval(
            gray, diameter=diameter, channels=[0, 0]
        )
        masks = masks.astype(np.int32)
        diam = float(diams) if np.isscalar(diams) else float(diams[0])

        if self.shape_filter:
            masks, stats = filter_by_shape(masks)
            self.last_filter_stats = stats
        else:
            self.last_filter_stats = None

        return masks, diam

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "engine": "cellpose",
            "model": self.model_type,
            "params": {
                "use_gpu": self.use_gpu,
                "shape_filter": self.shape_filter,
            },
        }
        if self.shape_filter:
            info["params"]["shape_filter_thresholds"] = {
                "max_eccentricity": MAX_ECCENTRICITY,
                "min_solidity":     MIN_SOLIDITY,
                "min_area_px":      MIN_AREA_PX,
                "max_area_frac":    MAX_AREA_FRAC,
            }
        if self.last_filter_stats is not None:
            info["filter_stats"] = self.last_filter_stats
        return info
