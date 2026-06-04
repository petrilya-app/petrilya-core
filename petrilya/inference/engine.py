"""Pluggable segmentation engines for Petrilya.

Three engines ship today:

* ``classical`` — pure scikit-image / OpenCV pipeline (Otsu + watershed).
  No ML, no model download, runs in tens of milliseconds. Recommended
  default for colony counting on Petri dish photos.

* ``cellpose-onnx`` — runs the bundled cellpose cyto3 ONNX weights
  (downloaded from huggingface in advance, not from cellpose.org) and
  delegates post-processing to ``cellpose.dynamics.compute_masks``.
  Heavier and slower but handles weird shapes / clusters better.

* ``cellpose`` — original cellpose package using PyTorch. Requires the
  cyto3 weights to be present in the local cellpose cache; if they
  aren't and the cellpose.org server is unreachable the engine will
  error out on first use. Kept for completeness.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Engine registry
# ---------------------------------------------------------------------

_REGISTRY: dict[str, type["Engine"]] = {}


def register(name: str):
    def deco(cls: type["Engine"]):
        _REGISTRY[name] = cls
        cls.engine_name = name
        return cls

    return deco


def available_engines() -> list[str]:
    return list(_REGISTRY.keys())


def build_engine(name: str, **kwargs: Any) -> "Engine":
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown engine {name!r}. Available: {available_engines()}"
        )
    return _REGISTRY[name](**kwargs)


# ---------------------------------------------------------------------
# Shared preprocessing
# ---------------------------------------------------------------------


def _find_dish_roi(
    image: np.ndarray,
    *,
    min_radius_frac: float = 0.15,
    max_radius_frac: float = 0.48,
) -> tuple[int, int, int] | None:
    """Find the Petri dish in an image via Hough circles.

    Returns ``(cx, cy, r)`` in pixel coordinates or ``None`` if no
    convincing circle was found.
    """
    from skimage import feature, transform
    from skimage.util import img_as_ubyte

    # Work on a downsampled copy — Hough is O(R * N), and we don't need
    # full resolution to find a big circle.
    scale = 1.0
    work = image
    h0, w0 = image.shape[:2]
    target = 600
    if max(h0, w0) > target:
        scale = target / max(h0, w0)
        new_h = int(h0 * scale)
        new_w = int(w0 * scale)
        from skimage.transform import resize

        work = (resize(image, (new_h, new_w), preserve_range=True, anti_aliasing=True)
                .astype(np.uint8))
    h, w = work.shape[:2]

    edges = feature.canny(img_as_ubyte(work), sigma=2.5)
    r_min = max(20, int(min(h, w) * min_radius_frac))
    r_max = max(r_min + 5, int(min(h, w) * max_radius_frac))
    radii = np.arange(r_min, r_max, max(2, (r_max - r_min) // 30))
    if len(radii) == 0:
        return None
    hough = transform.hough_circle(edges, radii)
    accums, cxs, cys, rads = transform.hough_circle_peaks(
        hough, radii, total_num_peaks=1
    )
    if len(rads) == 0 or accums[0] < 0.10:
        return None

    cx = int(cxs[0] / scale)
    cy = int(cys[0] / scale)
    r  = int(rads[0] / scale)
    return cx, cy, r


def _crop_to_dish(
    image: np.ndarray,
    roi: tuple[int, int, int] | None,
    *,
    inner_frac: float = 0.92,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop to the dish, masking everything outside the agar disc to 0.

    Returns ``(cropped_image, (x0, y0))`` so callers can paste masks
    back into the original image coordinates.
    """
    if roi is None:
        return image, (0, 0)

    cx, cy, r = roi
    inner = int(r * inner_frac)
    h, w = image.shape[:2]
    x0 = max(0, cx - r)
    y0 = max(0, cy - r)
    x1 = min(w, cx + r)
    y1 = min(h, cy + r)

    cropped = image[y0:y1, x0:x1].copy()
    ch, cw = cropped.shape[:2]
    # Mask outside the inner agar circle to "background" so the
    # downstream threshold doesn't latch onto the rim.
    yy, xx = np.ogrid[:ch, :cw]
    new_cx = cx - x0
    new_cy = cy - y0
    mask = (xx - new_cx) ** 2 + (yy - new_cy) ** 2 > inner * inner
    if cropped.ndim == 2:
        cropped[mask] = 0
    else:
        cropped[mask, :] = 0
    return cropped, (x0, y0)


def _uncrop_masks(
    masks_local: np.ndarray,
    offset: tuple[int, int],
    full_shape: tuple[int, int],
) -> np.ndarray:
    """Paste local masks back into a full-size label image."""
    full = np.zeros(full_shape, dtype=np.int32)
    x0, y0 = offset
    h, w = masks_local.shape
    full[y0 : y0 + h, x0 : x0 + w] = masks_local
    return full


# ---------------------------------------------------------------------
# Engine ABC
# ---------------------------------------------------------------------


EngineResult = tuple[np.ndarray, float]


class Engine(ABC):
    engine_name: str = "abstract"

    @abstractmethod
    def segment(self, image: np.ndarray, diameter: float | None = None) -> EngineResult:
        ...

    def describe(self) -> dict[str, Any]:
        return {"engine": self.engine_name}


# ---------------------------------------------------------------------
# Classical CV engine
# ---------------------------------------------------------------------


@register("classical")
class ClassicalEngine(Engine):
    """Top-hat + watershed colony counter, scoped to the detected dish ROI."""

    # Work crop is downscaled so the longest edge is at most this many
    # pixels — top-hat morphology is O(N * structuring-element-area)
    # and gets painful on 3K+ crops. Labels are upscaled back via
    # nearest-neighbour so IDs aren't blended.
    MAX_EDGE = 900

    def __init__(
        self,
        *,
        min_colony_px: int = 18,
        max_colony_px: int | None = None,
        polarity: str = "auto",
        smoothing_sigma: float = 1.2,
        use_gpu: bool = False,
        **_: Any,
    ) -> None:
        self.min_colony_px = int(min_colony_px)
        self.max_colony_px = max_colony_px
        self.polarity = polarity
        self.smoothing_sigma = float(smoothing_sigma)

    def segment(self, image, diameter=None):
        """Classical pipeline:

        1. Hough-circle to find the Petri dish; crop to it and mask out
           the rim so nothing outside the agar can be counted.
        2. Morphological top-hat in both polarities (bright spots on
           dark agar / dark spots on bright agar) — robust against the
           "auto polarity" trap that Otsu falls into when an image has
           a strong dark border or marker-pen text on the agar.
        3. Pick whichever top-hat has the stronger overall response —
           that's where the actual colonies are.
        4. Threshold the top-hat, label, watershed-split clusters.
        """
        from scipy import ndimage as ndi
        from skimage import filters, morphology, segmentation
        from skimage.feature import peak_local_max

        # Smart RGB→grayscale that preserves YELLOW/ORANGE/RED colonies on
        # cream agar by combining luminance with chroma. For grayscale
        # input this is a no-op.
        image = _to_intensity(image, mode="auto")
        original_shape = image.shape

        roi = _find_dish_roi(image)
        crop_full, offset = _crop_to_dish(image, roi)
        h0, w0 = crop_full.shape

        # Downsample the crop for the expensive morphology steps.
        if max(h0, w0) > self.MAX_EDGE:
            from skimage.transform import resize as _resize

            scale = self.MAX_EDGE / max(h0, w0)
            nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
            crop = (
                _resize(crop_full, (nh, nw), preserve_range=True, anti_aliasing=True)
                .astype(np.uint8)
            )
        else:
            scale = 1.0
            crop = crop_full

        in_dish = crop > 0 if roi is not None else np.ones_like(crop, bool)
        if in_dish.sum() < 100:
            return np.zeros(original_shape, np.int32), 0.0

        # Structuring element radius for top-hat (in WORK crop coords).
        # Must be LARGER than the biggest expected colony radius —
        # otherwise large colonies fit inside the disk and survive the
        # opening, so top-hat returns zero for them and the engine ends
        # up only finding tiny artefacts like marker-pen text strokes.
        if diameter is not None and diameter > 0:
            tophat_r = max(8, int(diameter * scale / 2.0 * 1.5))
        elif roi is not None:
            # ~8% of dish radius — covers colonies up to ~16% of the dish.
            tophat_r = max(12, int(roi[2] * scale * 0.08))
        else:
            tophat_r = max(12, int(min(crop.shape) * 0.04))
        # Cap so morphology stays affordable. With MAX_EDGE=900, disk(45)
        # area is ~6300 px and morphology completes in seconds.
        tophat_r = min(tophat_r, 45)

        img8 = crop  # already uint8

        # Smooth a little to suppress single-pixel noise.
        if self.smoothing_sigma > 0:
            img8 = ndi.gaussian_filter(img8, self.smoothing_sigma).astype(np.uint8)

        selem = morphology.disk(tophat_r)
        wt = morphology.white_tophat(img8, selem)  # bright spots
        bt = morphology.black_tophat(img8, selem)  # dark spots
        wt[~in_dish] = 0
        bt[~in_dish] = 0

        if self.polarity == "bright":
            response = wt
        elif self.polarity == "dark":
            response = bt
        else:
            # Auto: whichever top-hat has more contrast (sum of top-1%
            # of pixels) wins. This is empirically more robust than
            # comparing maxima alone.
            wt_score = float(np.percentile(wt[in_dish], 99))
            bt_score = float(np.percentile(bt[in_dish], 99))
            response = wt if wt_score >= bt_score else bt

        if response.max() < 5:
            return np.zeros(original_shape, np.int32), 0.0

        try:
            thresh = filters.threshold_otsu(response[response > 0])
        except Exception:
            thresh = max(int(response.max() * 0.25), 5)

        binary = (response > thresh) & in_dish
        binary = morphology.remove_small_objects(binary, min_size=self.min_colony_px)
        binary = ndi.binary_fill_holes(binary)

        if not binary.any():
            return np.zeros(original_shape, np.int32), 0.0

        dist = ndi.distance_transform_edt(binary)
        if dist.max() <= 0:
            return np.zeros(original_shape, np.int32), 0.0

        if diameter is None:
            est_radius = float(np.median(dist[binary])) * 2.0
            est_radius = max(3.0, est_radius)
        else:
            est_radius = float(diameter) / 2.0
        peak_min_dist = max(3, int(est_radius * 0.8))

        coords = peak_local_max(
            dist,
            min_distance=peak_min_dist,
            labels=binary.astype(np.int32),
        )
        if len(coords) == 0:
            return np.zeros(original_shape, np.int32), 0.0

        markers = np.zeros(dist.shape, dtype=np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
        markers = morphology.dilation(markers, morphology.disk(1))
        labels = segmentation.watershed(-dist, markers, mask=binary)

        # Shape filtering: keep ONLY round/convex regions. Marker text,
        # cracks in the agar, scuff marks etc. all have high eccentricity
        # or low solidity. Real colonies are nearly round.
        from skimage import measure

        for prop in measure.regionprops(labels):
            drop = False
            if prop.eccentricity > 0.85:
                drop = True
            if prop.solidity < 0.75:
                drop = True
            if self.max_colony_px is not None and prop.area > self.max_colony_px:
                drop = True
            # Reject tiny specks too — they're below what a colony can be.
            if prop.area < self.min_colony_px:
                drop = True
            if drop:
                labels[labels == prop.label] = 0

        labels = segmentation.relabel_sequential(labels)[0].astype(np.int32)

        # Upscale labels back to the full crop resolution (nearest-neighbour
        # so we don't blend IDs), then paste into the full image canvas.
        if scale != 1.0:
            from skimage.transform import resize as _resize

            labels_full_crop = (
                _resize(labels, (h0, w0), order=0, preserve_range=True,
                        anti_aliasing=False)
                .astype(np.int32)
            )
        else:
            labels_full_crop = labels
        full_labels = _uncrop_masks(labels_full_crop, offset, original_shape)
        return full_labels, (est_radius * 2.0) / max(scale, 1e-6)

    def describe(self):
        return {
            "engine": "classical",
            "params": {
                "min_colony_px": self.min_colony_px,
                "max_colony_px": self.max_colony_px,
                "polarity": self.polarity,
                "smoothing_sigma": self.smoothing_sigma,
            },
        }


# ---------------------------------------------------------------------
# Cellpose ONNX engine
# ---------------------------------------------------------------------


def _percentile_normalize(img: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    a, b = np.percentile(img, [lo, hi])
    if b - a < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    out = (img - a) / (b - a)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _to_intensity(image: np.ndarray, mode: str = "auto") -> np.ndarray:
    """Convert a 2D/RGB(A) image to a single-channel ``uint8`` intensity map
    that maximises colony contrast.

    Standard luminance (0.299 R + 0.587 G + 0.114 B) loses YELLOW colonies
    on a CREAM agar — both have similar luminance, the difference is almost
    entirely in the blue channel. We get a much better signal by combining
    the L channel with the magnitude of the LAB chroma, so any saturated
    colour (yellow, orange, red, pink) stands out from the near-neutral
    agar regardless of its luminance.

    ``mode``:
      ``"auto"``       — L + |chroma| (default; recommended for classical)
      ``"luminance"``  — plain rec.709-ish luminance (use for cellpose
                          since it was trained on it)
    """
    if image.ndim == 2:
        return image.astype(np.uint8, copy=False)

    if image.ndim == 3 and image.shape[-1] >= 3:
        rgb = image[..., :3].astype(np.float32)
        # rec.709 luminance
        Y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

        if mode == "luminance":
            return np.clip(Y, 0, 255).astype(np.uint8)

        # Cheap chroma estimate without doing a full LAB conversion:
        # max(R,G,B) - min(R,G,B) is the HSV saturation×value product,
        # which lights up exactly when the colour is saturated.
        mx = rgb.max(axis=-1)
        mn = rgb.min(axis=-1)
        chroma = mx - mn

        # Blend luminance with a SMALL chroma boost. Earlier we used a
        # heavy 0.6 multiplier — but colored marker text on the agar then
        # outshines real (low-chroma) colonies. With 0.25, saturated
        # yellow/orange colonies still gain some contrast but ink strokes
        # don't drown out the real signal.
        out = Y + 0.25 * chroma
        return np.clip(out, 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported image shape: {image.shape}")


@register("cellpose-onnx")
class CellposeOnnxEngine(Engine):
    """Cellpose cyto3 weights via ONNX Runtime + cellpose.dynamics."""

    DEFAULT_MODEL = "models/cyto3-fp16.onnx"
    # Max edge length fed to the network. Above this we resize down to
    # keep memory bounded (the network is dense — a 4000x3000 image needs
    # ~3.5 GB of activations at FP32).
    MAX_EDGE = 1024

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        use_gpu: bool = False,
        cellprob_threshold: float = 0.0,
        flow_threshold: float = 0.4,
        **_: Any,
    ) -> None:
        import onnxruntime as ort

        repo_root = Path(__file__).resolve().parents[2]
        self.model_path = Path(model_path) if model_path else repo_root / self.DEFAULT_MODEL
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Cellpose ONNX weights not found at {self.model_path}. "
                "Download cyto3-fp16.onnx from "
                "https://huggingface.co/kmlyyll/cellpose-cyto3-onnx "
                "and place it at models/cyto3-fp16.onnx."
            )

        providers = []
        if use_gpu:
            avail = ort.get_available_providers()
            if "CUDAExecutionProvider" in avail:
                providers.append("CUDAExecutionProvider")
            elif "CoreMLExecutionProvider" in avail:
                providers.append("CoreMLExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.cellprob_threshold = cellprob_threshold
        self.flow_threshold = flow_threshold
        self.use_gpu = use_gpu

    def segment(self, image, diameter=None):
        from skimage.transform import resize

        # Cellpose was trained on luminance — use plain Y for fidelity.
        image = _to_intensity(image, mode="luminance")
        original_shape = image.shape

        # Crop to dish to feed only the agar region to the net.
        roi = _find_dish_roi(image)
        crop, offset = _crop_to_dish(image, roi)
        h0, w0 = crop.shape

        # Resize crop so longest edge <= MAX_EDGE
        if max(h0, w0) > self.MAX_EDGE:
            scale = self.MAX_EDGE / max(h0, w0)
            new_h = int(round(h0 * scale))
            new_w = int(round(w0 * scale))
            small = (
                resize(crop, (new_h, new_w), preserve_range=True, anti_aliasing=True)
                .astype(np.float32)
            )
        else:
            scale = 1.0
            small = crop.astype(np.float32)

        norm = _percentile_normalize(small)
        x = np.stack([norm, np.zeros_like(norm)], axis=0)[None].astype(np.float32)

        out = self.session.run(None, {self.input_name: x})[0][0]
        flow_y = out[0]
        flow_x = out[1]
        cellprob = out[2]
        dP = np.stack([flow_y, flow_x], axis=0)

        from cellpose import dynamics

        # cellpose 3.1.x returns just the label array (older versions
        # returned (labels, p)); be defensive.
        result = dynamics.compute_masks(
            dP,
            cellprob,
            cellprob_threshold=self.cellprob_threshold,
            flow_threshold=self.flow_threshold,
            interp=True,
        )
        labels_small = result[0] if isinstance(result, tuple) else result
        labels_small = labels_small.astype(np.int32)

        # Upscale labels back to the crop resolution (nearest-neighbour
        # so we don't blend label IDs).
        if scale != 1.0:
            labels_crop = (
                resize(labels_small, (h0, w0), order=0, preserve_range=True,
                       anti_aliasing=False)
                .astype(np.int32)
            )
        else:
            labels_crop = labels_small

        full_labels = _uncrop_masks(labels_crop, offset, original_shape)
        if full_labels.max() > 0:
            from skimage import measure

            areas = np.array([p.area for p in measure.regionprops(full_labels)])
            est_diam = float(2.0 * np.sqrt(np.median(areas) / np.pi))
        else:
            est_diam = 0.0
        return full_labels, est_diam

    def describe(self):
        return {
            "engine": "cellpose-onnx",
            "model": "cyto3-fp16.onnx",
            "params": {
                "use_gpu": self.use_gpu,
                "cellprob_threshold": self.cellprob_threshold,
                "flow_threshold": self.flow_threshold,
                "max_edge": self.MAX_EDGE,
            },
        }


# ---------------------------------------------------------------------
# Original PyTorch Cellpose engine
# ---------------------------------------------------------------------


@register("cellpose")
class CellposeEngine(Engine):
    """Original cellpose package using PyTorch weights.

    Requires cellpose's weight cache to be populated.
    """

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

    def segment(self, image, diameter=None):
        image = _to_intensity(image, mode="luminance")
        masks, _flows, _styles, diams = self.model.eval(
            image, diameter=diameter, channels=[0, 0]
        )
        masks = masks.astype(np.int32)
        diam = float(diams) if np.isscalar(diams) else float(diams[0])
        return masks, diam

    def describe(self):
        return {
            "engine": "cellpose",
            "model": self.model_type,
            "params": {"use_gpu": self.use_gpu},
        }
