"""Mock segmentation engine for UI development before cellpose.org is back."""

from __future__ import annotations

import time

import numpy as np
from skimage.draw import disk


def mock_segment(
    image: np.ndarray,
    n_colonies: int = 120,
    delay_seconds: float = 1.5,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Generate fake colony masks with realistic-looking blobs.

    Imitates a real inference run so the UI can be tested end-to-end
    while the real Cellpose model weights are unavailable.
    """
    time.sleep(delay_seconds)
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    masks = np.zeros((h, w), dtype=np.int32)

    margin = 40
    min_r, max_r = 8, 22
    placed = 0
    attempts = 0
    max_attempts = n_colonies * 20

    while placed < n_colonies and attempts < max_attempts:
        attempts += 1
        cy = rng.integers(margin, h - margin)
        cx = rng.integers(margin, w - margin)
        r = rng.integers(min_r, max_r)
        rr, cc = disk((cy, cx), r, shape=(h, w))
        # avoid overlap
        if masks[rr, cc].any():
            continue
        placed += 1
        masks[rr, cc] = placed

    return masks, float(max_r * 2)
