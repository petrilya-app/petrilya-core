"""Image canvas with zoom, pan, mask overlay, and editing.

QGraphicsView-based canvas. Mouse wheel zooms around the cursor.
Hold Space (or middle mouse button) to pan. In edit modes, the left
mouse button erases or paints masks.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QSizePolicy,
)


def _numpy_to_qpixmap(arr: np.ndarray) -> QPixmap:
    """Convert a 2D grayscale or 3D RGB(A) numpy array to a QPixmap."""
    if arr.ndim == 2:
        arr = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        h, w = arr.shape
        qimg = QImage(arr.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    elif arr.ndim == 3:
        arr = np.ascontiguousarray(arr.astype(np.uint8, copy=False))
        h, w, c = arr.shape
        if c == 3:
            qimg = QImage(arr.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        elif c == 4:
            qimg = QImage(arr.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888)
        else:
            raise ValueError(f"Unsupported channel count: {c}")
    else:
        raise ValueError(f"Unsupported image ndim: {arr.ndim}")
    return QPixmap.fromImage(qimg.copy())


# Back-compat alias for any external caller
_numpy_to_qpixmap_gray = _numpy_to_qpixmap


def _masks_to_overlay_pixmap(masks: np.ndarray, alpha: int = 110) -> QPixmap:
    h, w = masks.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    if masks.max() > 0:
        rng = np.random.default_rng(0)
        n = int(masks.max()) + 1
        palette = rng.integers(40, 230, size=(n, 3), dtype=np.uint8)
        palette[0] = 0
        rgb = palette[masks]
        rgba[..., :3] = rgb
        rgba[..., 3] = np.where(masks > 0, alpha, 0).astype(np.uint8)
    rgba = np.ascontiguousarray(rgba)
    qimg = QImage(rgba.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class ImageCanvas(QGraphicsView):
    """Zoomable, pannable, editable image canvas."""

    EDIT_NONE = "none"
    EDIT_ERASE = "erase"
    EDIT_BRUSH = "brush"

    masks_edited = Signal()  # emitted after every user-triggered mask change

    def __init__(self) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QColor(20, 20, 20))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

        self._base_item: QGraphicsPixmapItem | None = None
        self._overlay_item: QGraphicsPixmapItem | None = None
        self._masks: np.ndarray | None = None
        self._alpha: int = 110
        self._edit_mode: str = self.EDIT_NONE
        self._brush_size: int = 14
        self._brush_active_label: int = 0
        self._space_held: bool = False

        self._placeholder = self._scene.addText(
            "Drag and drop an image here\nor use File > Open"
        )
        self._placeholder.setDefaultTextColor(QColor(140, 140, 140))

    # ---------------------------- public api --------------------------------

    def set_image(self, image: np.ndarray) -> None:
        self._scene.clear()
        self._placeholder = None
        self._base_item = QGraphicsPixmapItem(_numpy_to_qpixmap(image))
        self._scene.addItem(self._base_item)
        self._overlay_item = None
        self._masks = None
        self._scene.setSceneRect(self._base_item.boundingRect())
        self.fit_to_window()

    def set_masks(self, masks: np.ndarray) -> None:
        self._masks = masks.astype(np.int32, copy=True)
        self._refresh_overlay()

    def clear_masks(self) -> None:
        self._masks = None
        if self._overlay_item is not None:
            self._scene.removeItem(self._overlay_item)
            self._overlay_item = None

    def masks(self) -> np.ndarray | None:
        return self._masks

    def set_overlay_alpha(self, alpha: int) -> None:
        self._alpha = max(0, min(255, alpha))
        self._refresh_overlay()

    def set_edit_mode(self, mode: str) -> None:
        if mode not in (self.EDIT_NONE, self.EDIT_ERASE, self.EDIT_BRUSH):
            raise ValueError(f"Unknown edit mode: {mode}")
        self._edit_mode = mode
        if mode == self.EDIT_NONE:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        else:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def set_brush_size(self, size_px: int) -> None:
        self._brush_size = max(1, int(size_px))

    def fit_to_window(self) -> None:
        if self._base_item is None:
            return
        self.fitInView(self._base_item, Qt.AspectRatioMode.KeepAspectRatio)

    def reset_zoom(self) -> None:
        self.resetTransform()

    # ----------------------------- rendering --------------------------------

    def _refresh_overlay(self) -> None:
        if self._base_item is None or self._masks is None:
            return
        pix = _masks_to_overlay_pixmap(self._masks, alpha=self._alpha)
        if self._overlay_item is None:
            self._overlay_item = QGraphicsPixmapItem(pix)
            self._overlay_item.setZValue(1)
            self._scene.addItem(self._overlay_item)
        else:
            self._overlay_item.setPixmap(pix)

    # ----------------------------- interaction ------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 (Qt API)
        if self._base_item is None:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.key() == Qt.Key.Key_F:
            self.fit_to_window()
            event.accept()
            return
        if event.key() == Qt.Key.Key_0:
            self.reset_zoom()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(
                Qt.CursorShape.CrossCursor
                if self._edit_mode != self.EDIT_NONE
                else Qt.CursorShape.ArrowCursor
            )
            event.accept()
            return
        super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        # middle button always pans
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            fake = QMouseEvent(
                event.type(),
                event.position(),
                Qt.MouseButton.LeftButton,
                event.buttons() | Qt.MouseButton.LeftButton,
                event.modifiers(),
            )
            super().mousePressEvent(fake)
            return

        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._space_held
            and self._masks is not None
            and self._edit_mode != self.EDIT_NONE
        ):
            self._apply_edit_at(self.mapToScene(event.position().toPoint()), starting=True)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and not self._space_held
            and self._masks is not None
            and self._edit_mode == self.EDIT_BRUSH
        ):
            self._apply_edit_at(
                self.mapToScene(event.position().toPoint()), starting=False
            )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._brush_active_label = 0
            self.masks_edited.emit()
        super().mouseReleaseEvent(event)

    # ---------------------------- editing logic -----------------------------

    def _apply_edit_at(self, scene_pos: QPointF, starting: bool) -> None:
        if self._masks is None:
            return
        x = int(scene_pos.x())
        y = int(scene_pos.y())
        h, w = self._masks.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        if self._edit_mode == self.EDIT_ERASE:
            label = int(self._masks[y, x])
            if label > 0:
                self._masks[self._masks == label] = 0
                self._refresh_overlay()
            return

        if self._edit_mode == self.EDIT_BRUSH:
            if starting:
                self._brush_active_label = int(self._masks.max()) + 1
            label = self._brush_active_label or int(self._masks.max()) + 1
            r = self._brush_size
            yy, xx = np.ogrid[:h, :w]
            disk = (yy - y) ** 2 + (xx - x) ** 2 <= r * r
            self._masks[disk] = label
            self._refresh_overlay()
