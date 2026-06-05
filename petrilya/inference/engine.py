"""Cellpose-based segmentation for Petrilya.

Uses Cellpose with the cyto3 model (weights auto-downloaded from
cellpose.org on first use and cached in ``~/.cellpose/models/``).

Two post-processing passes filter the raw Cellpose output, because
cyto3 was trained on microscopy of individual cells, not macro photos
of Petri dishes — so on a phone photo it ALSO segments marker-pen
text and rim artefacts as 'cells':

  1. Diameter hint. We detect the dish via Hough circles and pass
     Cellpose a ``diameter`` argument calibrated to a fraction of the
     dish radius. Without this hint Cellpose auto-scales to whatever
     looks cell-sized in the picture, which on a 4K photo is the
     pixels-of-the-text, not the actual colonies.

  2. Per-region filter on shape AND colour. Real colonies are nearly
     round, near-neutral cream/yellow, and large. Text strokes are
     elongated OR saturated OR dark OR all of the above. Filtering
     on those properties together drops almost everything that isn't
     an actual colony.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Filter thresholds (tuned on the AGAR sample set). All overridable.
# ---------------------------------------------------------------------
MAX_ECCENTRICITY = 0.85          # 0=circle, 1=line; text strokes >0.9.
MIN_SOLIDITY     = 0.80          # area / convex-hull area.
MIN_AREA_PX      = 150           # min sane colony area on a 1-3K image.
MAX_AREA_FRAC    = 0.20          # one 'colony' > 20% frame is wrong.
MAX_SATURATION   = 0.45          # 0..1, marker pen typically >0.5.
MIN_LUMA         = 55            # 0..255, anything darker is text/shadow.


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Return a ``uint8`` grayscale view of a 2D or RGB(A) input."""
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)
    if image.ndim == 3 and image.shape[-1] >= 3:
        rgb = image[..., :3].astype(np.float32)
        Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        return np.clip(Y, 0, 255).astype(np.uint8)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _estimate_dish_radius(image_gray: np.ndarray) -> int | None:
    """Hough-circle dish detector. Returns radius in pixels or None."""
    from skimage import feature, transform
    from skimage.transform import resize

    h, w = image_gray.shape
    target = 600
    if max(h, w) > target:
        scale = target / max(h, w)
        small = (
            resize(image_gray, (int(h * scale), int(w * scale)),
                   preserve_range=True, anti_aliasing=True)
            .astype(np.uint8)
        )
    else:
        scale = 1.0
        small = image_gray
    edges = feature.canny(small, sigma=2.5)
    sh, sw = small.shape
    r_min = max(20, int(min(sh, sw) * 0.15))
    r_max = max(r_min + 5, int(min(sh, sw) * 0.48))
    radii = np.arange(r_min, r_max, max(2, (r_max - r_min) // 30))
    if len(radii) == 0:
        return None
    hough = transform.hough_circle(edges, radii)
    accums, _, _, rads = transform.hough_circle_peaks(
        hough, radii, total_num_peaks=1
    )
    if len(rads) == 0 or accums[0] < 0.10:
        return None
    return int(rads[0] / scale)


def filter_by_shape_and_colour(
    masks: np.ndarray,
    image: np.ndarray | None = None,
    *,
    max_eccentricity: float = MAX_ECCENTRICITY,
    min_solidity: float = MIN_SOLIDITY,
    min_area_px: int = MIN_AREA_PX,
    max_area_frac: float = MAX_AREA_FRAC,
    max_saturation: float = MAX_SATURATION,
    min_luma: float = MIN_LUMA,
) -> tuple[np.ndarray, dict[str, int]]:
    """Drop regions that don't look like real Petri-dish colonies.

    ``image`` is optional and only used when colour information is
    available — saturation / luminance gates are skipped for grayscale
    input.
    """
    from skimage import measure, segmentation

    stats = {
        "before": int(masks.max()),
        "dropped_eccentricity": 0,
        "dropped_solidity": 0,
        "dropped_area": 0,
        "dropped_saturation": 0,
        "dropped_luma": 0,
    }
    if masks.max() == 0:
        stats["kept"] = 0
        return masks.astype(np.int32, copy=False), stats

    out = masks.astype(np.int32, copy=True)
    image_area = float(masks.size)
    max_area_px = int(image_area * max_area_frac)

    rgb = None
    if image is not None and image.ndim == 3 and image.shape[-1] >= 3:
        rgb = image[..., :3].astype(np.float32)

    for prop in measure.regionprops(out):
        reason = None
        if prop.eccentricity > max_eccentricity:
            reason = "dropped_eccentricity"
        elif prop.solidity < min_solidity:
            reason = "dropped_solidity"
        elif prop.area < min_area_px or prop.area > max_area_px:
            reason = "dropped_area"
        elif rgb is not None:
            ys, xs = np.where(out == prop.label)
            px = rgb[ys, xs]                  # (N, 3)
            mx = px.max(axis=1)
            mn = px.min(axis=1)
            sat = (mx - mn) / (mx + 1.0)      # 0..1 normalised saturation
            luma = 0.2126 * px[:, 0] + 0.7152 * px[:, 1] + 0.0722 * px[:, 2]
            if float(np.median(sat)) > max_saturation:
                reason = "dropped_saturation"
            elif float(np.median(luma)) < min_luma:
                reason = "dropped_luma"

        if reason:
            out[out == prop.label] = 0
            stats[reason] += 1

    out = segmentation.relabel_sequential(out)[0].astype(np.int32)
    stats["kept"] = int(out.max())
    return out, stats


class CellposeEngine:
    """Cellpose + dish-aware diameter hint + shape/colour post-filter."""

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
        self.last_diameter: float | None = None

    def segment(
        self,
        image: np.ndarray,
        diameter: float | None = None,
    ) -> tuple[np.ndarray, float]:
        gray = to_grayscale(image)

        # If the caller didn't pin a diameter, try to estimate one from
        # the dish radius — colonies are typically 3-8% of the dish
        # diameter on a phone photo. Without this hint Cellpose auto-
        # scales to whatever looks cell-sized in the photo, which on a
        # 4K image of a plate with marker-pen text is the individual
        # character pixels — disaster.
        if diameter is None:
            dish_r = _estimate_dish_radius(gray)
            if dish_r is not None:
                # 5% of dish diameter ≈ typical bacterial colony
                diameter = max(15.0, 2.0 * dish_r * 0.05)

        self.last_diameter = diameter

        masks, _flows, _styles, diams = self.model.eval(
            gray, diameter=diameter, channels=[0, 0]
        )
        masks = masks.astype(np.int32)
        diam_out = float(diams) if np.isscalar(diams) else float(diams[0])

        if self.shape_filter:
            masks, stats = filter_by_shape_and_colour(masks, image)
            self.last_filter_stats = stats
        else:
            self.last_filter_stats = None

        return masks, diam_out

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "engine": "cellpose",
            "model": self.model_type,
            "params": {
                "use_gpu": self.use_gpu,
                "shape_filter": self.shape_filter,
                "diameter_hint": self.last_diameter,
            },
        }
        if self.shape_filter:
            info["params"]["shape_filter_thresholds"] = {
                "max_eccentricity": MAX_ECCENTRICITY,
                "min_solidity":     MIN_SOLIDITY,
                "min_area_px":      MIN_AREA_PX,
                "max_area_frac":    MAX_AREA_FRAC,
                "max_saturation":   MAX_SATURATION,
                "min_luma":         MIN_LUMA,
            }
        if self.last_filter_stats is not None:
            info["filter_stats"] = self.last_filter_stats
        return info
