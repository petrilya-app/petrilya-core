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
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QSizePolicy,
)
from PySide6.QtCore import QRectF
from PySide6.QtGui import QBrush, QPen


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


class _RevealOverlayItem(QGraphicsPixmapItem):
    """Pixmap item that paints only its LEFT ``reveal`` fraction.

    Lets the canvas implement a Lightroom-style before/after split: at
    reveal=0 nothing painted (pure 'before'), at 1.0 the entire overlay
    is painted (pure 'after'), in between the user sees the original on
    the right and the analysed result on the left.
    """

    def __init__(self, pixmap):
        super().__init__(pixmap)
        self._reveal = 1.0

    def set_reveal(self, fraction: float) -> None:
        self._reveal = max(0.0, min(1.0, float(fraction)))
        self.update()

    def paint(self, painter, option, widget=None):  # noqa: N802
        if self._reveal <= 0.0:
            return
        if self._reveal >= 1.0:
            super().paint(painter, option, widget)
            return
        pix = self.pixmap()
        split_x = pix.width() * self._reveal
        painter.save()
        painter.setClipRect(QRectF(0, 0, split_x, pix.height()))
        super().paint(painter, option, widget)
        painter.restore()


class ImageCanvas(QGraphicsView):
    """Zoomable, pannable, editable image canvas."""

    EDIT_NONE = "none"
    EDIT_ERASE = "erase"
    EDIT_BRUSH = "brush"

    # ROI dragging modes
    ROI_NONE = "none"
    ROI_MOVE = "move"
    ROI_RESIZE = "resize"
    ROI_HIT_RADIUS_SCENE = 18  # in scene units; pickable handle size

    masks_edited = Signal()  # emitted after every user-triggered mask change
    roi_changed = Signal()    # emitted when the dish ROI is moved/resized

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
        self._overlay_item: _RevealOverlayItem | None = None
        self._masks: np.ndarray | None = None
        self._alpha: int = 110
        self._reveal: float = 1.0
        self._edit_mode: str = self.EDIT_NONE
        self._brush_size: int = 14
        self._brush_active_label: int = 0
        self._space_held: bool = False

        # Dish ROI (manual override of the engine's auto-detect)
        self._roi_circle_item: QGraphicsEllipseItem | None = None
        self._roi_center_handle: QGraphicsEllipseItem | None = None
        self._roi_edge_handle: QGraphicsEllipseItem | None = None
        self._roi_visible: bool = False
        self._roi: tuple[float, float, float] | None = None  # (cx, cy, r)
        self._roi_drag_mode: str = self.ROI_NONE
        self._roi_drag_offset: tuple[float, float] = (0.0, 0.0)

        self._placeholder = self._scene.addText(
            "Drag and drop an image here\nor use File > Open"
        )
        self._placeholder.setDefaultTextColor(QColor(140, 140, 140))

    # ---------------------------- public api --------------------------------

    def set_image(self, image: np.ndarray) -> None:
        self._scene.clear()
        self._placeholder = None
        self._roi_circle_item = None
        self._roi_center_handle = None
        self._roi_edge_handle = None
        self._base_item = QGraphicsPixmapItem(_numpy_to_qpixmap(image))
        self._scene.addItem(self._base_item)
        self._overlay_item = None
        self._masks = None
        self._scene.setSceneRect(self._base_item.boundingRect())

        # Default ROI = centred circle covering ~80% of the shorter side
        h, w = image.shape[:2]
        side = min(h, w)
        self._roi = (w / 2, h / 2, side * 0.40)
        if self._roi_visible:
            self._draw_roi()

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

    # ----------------------------- reveal split -----------------------------

    def set_reveal(self, fraction: float) -> None:
        """0 = pure 'before' (no overlay shown), 1 = pure 'after'."""
        self._reveal = max(0.0, min(1.0, float(fraction)))
        if self._overlay_item is not None:
            self._overlay_item.set_reveal(self._reveal)

    # ----------------------------- dish ROI ---------------------------------

    def set_roi_visible(self, visible: bool) -> None:
        self._roi_visible = bool(visible)
        if not visible:
            self._clear_roi_items()
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        elif self._roi is not None:
            self._draw_roi()

    def is_roi_visible(self) -> bool:
        return self._roi_visible

    def set_dish_roi(self, cx: float, cy: float, r: float) -> None:
        self._roi = (float(cx), float(cy), max(10.0, float(r)))
        if self._roi_visible:
            self._draw_roi()

    def dish_roi(self) -> tuple[float, float, float] | None:
        return self._roi

    def _clear_roi_items(self) -> None:
        for item in (self._roi_circle_item, self._roi_center_handle, self._roi_edge_handle):
            if item is not None and item.scene() is self._scene:
                self._scene.removeItem(item)
        self._roi_circle_item = None
        self._roi_center_handle = None
        self._roi_edge_handle = None

    def _draw_roi(self) -> None:
        if self._roi is None or self._base_item is None:
            return
        cx, cy, r = self._roi

        roi_pen = QPen(QColor(91, 142, 217), 0)
        roi_pen.setCosmetic(True)  # constant 2px regardless of zoom
        roi_pen.setWidth(2)
        roi_pen.setStyle(Qt.PenStyle.DashLine)
        roi_pen.setDashPattern([5, 5])

        handle_pen = QPen(QColor(91, 142, 217), 0)
        handle_pen.setCosmetic(True)
        handle_pen.setWidth(1)
        handle_brush = QBrush(QColor(255, 255, 255, 230))

        # Main outline
        if self._roi_circle_item is None:
            self._roi_circle_item = QGraphicsEllipseItem()
            self._roi_circle_item.setZValue(5)
            self._scene.addItem(self._roi_circle_item)
        self._roi_circle_item.setPen(roi_pen)
        self._roi_circle_item.setRect(cx - r, cy - r, 2 * r, 2 * r)

        # Handles — small filled circles in scene coords. Drawn at 12 scene px
        # which then scales with zoom; that's fine — bigger handle when
        # zoomed in is convenient.
        h = max(6.0, r * 0.04)

        if self._roi_center_handle is None:
            self._roi_center_handle = QGraphicsEllipseItem()
            self._roi_center_handle.setZValue(6)
            self._scene.addItem(self._roi_center_handle)
        self._roi_center_handle.setPen(handle_pen)
        self._roi_center_handle.setBrush(handle_brush)
        self._roi_center_handle.setRect(cx - h, cy - h, 2 * h, 2 * h)

        if self._roi_edge_handle is None:
            self._roi_edge_handle = QGraphicsEllipseItem()
            self._roi_edge_handle.setZValue(6)
            self._scene.addItem(self._roi_edge_handle)
        self._roi_edge_handle.setPen(handle_pen)
        self._roi_edge_handle.setBrush(handle_brush)
        self._roi_edge_handle.setRect(cx + r - h, cy - h, 2 * h, 2 * h)

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
            self._overlay_item = _RevealOverlayItem(pix)
            self._overlay_item.setZValue(1)
            self._overlay_item.set_reveal(self._reveal)
            self._scene.addItem(self._overlay_item)
        else:
            self._overlay_item.setPixmap(pix)
            self._overlay_item.set_reveal(self._reveal)

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

    def _roi_hit_test(self, scene_pos):
        if not self._roi_visible or self._roi is None:
            return self.ROI_NONE
        cx, cy, r = self._roi
        x, y = scene_pos.x(), scene_pos.y()
        # scene units; convert to a viewport-pixel-equivalent so the hit
        # area stays constant at every zoom level
        hit_scene = self.ROI_HIT_RADIUS_SCENE / max(self.transform().m11(), 1e-6)
        d_center = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
        if d_center <= hit_scene:
            return self.ROI_MOVE
        # near the circle outline?
        if abs(d_center - r) <= hit_scene:
            return self.ROI_RESIZE
        return self.ROI_NONE

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

        # ROI takes precedence over edit modes when visible and the click
        # is on one of the handles.
        if event.button() == Qt.MouseButton.LeftButton and not self._space_held:
            scene_pos = self.mapToScene(event.position().toPoint())
            hit = self._roi_hit_test(scene_pos)
            if hit != self.ROI_NONE:
                self._roi_drag_mode = hit
                cx, cy, _ = self._roi
                self._roi_drag_offset = (scene_pos.x() - cx, scene_pos.y() - cy)
                self.viewport().setCursor(
                    Qt.CursorShape.ClosedHandCursor if hit == self.ROI_MOVE
                    else Qt.CursorShape.SizeFDiagCursor
                )
                event.accept()
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
        # Handle ROI dragging first
        if self._roi_drag_mode != self.ROI_NONE and self._roi is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            x, y = scene_pos.x(), scene_pos.y()
            cx, cy, r = self._roi
            if self._roi_drag_mode == self.ROI_MOVE:
                ox, oy = self._roi_drag_offset
                self._roi = (x - ox, y - oy, r)
            else:  # ROI_RESIZE
                new_r = max(15.0, ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5)
                self._roi = (cx, cy, new_r)
            self._draw_roi()
            self.roi_changed.emit()
            event.accept()
            return

        # Hover cursor over ROI handles when not dragging
        if (
            self._roi_visible
            and self._roi_drag_mode == self.ROI_NONE
            and not (event.buttons() & Qt.MouseButton.LeftButton)
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            hit = self._roi_hit_test(scene_pos)
            if hit == self.ROI_MOVE:
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            elif hit == self.ROI_RESIZE:
                self.viewport().setCursor(Qt.CursorShape.SizeFDiagCursor)
            else:
                self.viewport().setCursor(
                    Qt.CursorShape.CrossCursor
                    if self._edit_mode != self.EDIT_NONE
                    else Qt.CursorShape.ArrowCursor
                )

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
        if self._roi_drag_mode != self.ROI_NONE:
            self._roi_drag_mode = self.ROI_NONE
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
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
