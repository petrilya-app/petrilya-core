"""Image canvas with mask overlay."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy


def numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert a 2D grayscale numpy array to QPixmap."""
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D grayscale, got shape {arr.shape}")
    arr = arr.astype(np.uint8, copy=False)
    h, w = arr.shape
    qimg = QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    return QPixmap.fromImage(qimg.copy())


def masks_to_overlay_pixmap(masks: np.ndarray, alpha: int = 110) -> QPixmap:
    """Render labeled masks as a colored RGBA overlay."""
    if masks.ndim != 2:
        raise ValueError("masks must be 2D")
    h, w = masks.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if masks.max() == 0:
        qimg = QImage(rgba.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(qimg.copy())

    # deterministic color per label
    rng = np.random.default_rng(0)
    n = int(masks.max()) + 1
    palette = rng.integers(40, 230, size=(n, 3), dtype=np.uint8)
    palette[0] = 0
    rgb = palette[masks]
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(masks > 0, alpha, 0).astype(np.uint8)

    qimg = QImage(rgba.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class ImageView(QLabel):
    """Displays an image with optional segmentation overlay."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "background:#1e1e1e; border:1px dashed #444; color:#888; font-size:14px;"
        )
        self.setText("Drag and drop an image here\nor use File > Open")
        self._base: QPixmap | None = None
        self._overlay: QPixmap | None = None
        self._show_overlay = True

    def set_image(self, image: np.ndarray) -> None:
        self._base = numpy_to_qpixmap(image)
        self._overlay = None
        self._render()

    def set_overlay(self, masks: np.ndarray) -> None:
        self._overlay = masks_to_overlay_pixmap(masks)
        self._render()

    def toggle_overlay(self, visible: bool) -> None:
        self._show_overlay = visible
        self._render()

    def clear_overlay(self) -> None:
        self._overlay = None
        self._render()

    def _render(self) -> None:
        if self._base is None:
            return
        composed = QPixmap(self._base.size())
        composed.fill(Qt.GlobalColor.transparent)
        p = QPainter(composed)
        p.drawPixmap(0, 0, self._base)
        if self._overlay is not None and self._show_overlay:
            p.drawPixmap(0, 0, self._overlay)
        p.end()
        scaled = composed.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        if self._base is not None:
            self._render()
