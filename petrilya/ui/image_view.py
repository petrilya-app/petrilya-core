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
    QGraphicsBlurEffect,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QSizePolicy,
)
from PySide6.QtCore import QPointF, QRectF
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
    """Pixmap item that paints only RIGHT of ``split_x`` (in pixmap coords).

    Lets the canvas implement a Lightroom-style before/after split:
    ``split_x`` is the x position (in image / pixmap pixels) of the
    vertical divider. Left of the divider the user sees the bare
    original photo (the base item shows through); right of the divider
    the analysed overlay is painted on top.

    When ``split_x`` is ``None`` we render the overlay over the entire
    pixmap (no split active — back to the regular full-overlay mode).
    """

    def __init__(self, pixmap):
        super().__init__(pixmap)
        self._split_x: float | None = None

    def set_split_x(self, x: float | None) -> None:
        if x is None:
            self._split_x = None
        else:
            pix_w = self.pixmap().width()
            self._split_x = max(0.0, min(float(pix_w), float(x)))
        self.update()

    def paint(self, painter, option, widget=None):  # noqa: N802
        if self._split_x is None:
            super().paint(painter, option, widget)
            return
        pix = self.pixmap()
        if self._split_x >= pix.width():
            return  # nothing to show on the right
        painter.save()
        painter.setClipRect(
            QRectF(self._split_x, 0, pix.width() - self._split_x, pix.height())
        )
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

    # Split-handle dragging
    SPLIT_HIT_RADIUS_VIEW = 28  # in viewport pixels — handle is a fixed size

    masks_edited = Signal()    # emitted after every user-triggered mask change
    roi_changed = Signal()     # emitted when the dish ROI is moved/resized
    split_changed = Signal()   # emitted when the before/after split is moved
    colony_clicked = Signal(int)  # emitted with the label id of a clicked colony

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

        # Before / after split — vertical line + draggable handle on the
        # image. split_x is in scene/pixmap coordinates.
        self._split_visible: bool = False
        self._split_x: float | None = None
        self._split_line_shadow_item: QGraphicsLineItem | None = None
        self._split_line_item: QGraphicsLineItem | None = None
        self._split_handle_shadow_item: QGraphicsEllipseItem | None = None
        self._split_handle_item: QGraphicsEllipseItem | None = None
        self._split_handle_inner: QGraphicsEllipseItem | None = None
        self._split_handle_ring_item: QGraphicsEllipseItem | None = None
        self._split_arrow_item = None
        self._split_dragging: bool = False

        # Table-row → canvas selection: highlight one colony in vivid
        # contrast over the regular overlay.
        self._highlight_label: int | None = None
        self._highlight_overlay_item: QGraphicsPixmapItem | None = None

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
        self._split_line_shadow_item = None
        self._split_line_item = None
        self._split_handle_shadow_item = None
        self._split_handle_item = None
        self._split_handle_inner = None
        self._split_handle_ring_item = None
        self._split_arrow_item = None
        self._highlight_overlay_item = None
        self._highlight_label = None
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

        # Default split at the horizontal midpoint of the image
        self._split_x = w / 2
        if self._split_visible:
            self._draw_split()

        self.fit_to_window()

    def set_masks(self, masks: np.ndarray) -> None:
        self._masks = masks.astype(np.int32, copy=True)
        self._refresh_overlay()
        # New mask set invalidates any previously highlighted label.
        self._highlight_label = None
        self._refresh_highlight()

    def clear_masks(self) -> None:
        self._masks = None
        if self._overlay_item is not None:
            self._scene.removeItem(self._overlay_item)
            self._overlay_item = None
        if self._highlight_overlay_item is not None:
            self._scene.removeItem(self._highlight_overlay_item)
            self._highlight_overlay_item = None
        self._highlight_label = None

    def masks(self) -> np.ndarray | None:
        return self._masks

    # ---- single-colony highlight (table-row → image link) ----

    def set_highlighted_label(self, label_id: int | None) -> None:
        """Highlight one colony in vivid magenta + white outline.

        Pass ``None`` (or 0) to clear the highlight.
        """
        if not label_id:
            self._highlight_label = None
        else:
            self._highlight_label = int(label_id)
        self._refresh_highlight()

    def highlighted_label(self) -> int | None:
        return self._highlight_label

    def _refresh_highlight(self) -> None:
        # Remove old highlight item
        if self._highlight_overlay_item is not None:
            if self._highlight_overlay_item.scene() is self._scene:
                self._scene.removeItem(self._highlight_overlay_item)
            self._highlight_overlay_item = None

        if (
            self._highlight_label is None
            or self._masks is None
            or self._base_item is None
        ):
            return

        mask = self._masks == self._highlight_label
        if not mask.any():
            return

        from scipy import ndimage as ndi

        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Vivid magenta tint over the colony (contrasts against typical
        # warm cream/yellow colonies, never collides with overlay palette).
        rgba[mask, 0] = 255
        rgba[mask, 1] = 30
        rgba[mask, 2] = 200
        rgba[mask, 3] = 110

        # Bright white outline (2 px dilation - original mask = ring)
        dilated = ndi.binary_dilation(mask, iterations=3)
        border = dilated & ~mask
        rgba[border, 0] = 255
        rgba[border, 1] = 255
        rgba[border, 2] = 255
        rgba[border, 3] = 255

        rgba = np.ascontiguousarray(rgba)
        qimg = QImage(rgba.tobytes(), w, h, 4 * w, QImage.Format.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg.copy())

        self._highlight_overlay_item = QGraphicsPixmapItem(pix)
        # Above the regular mask overlay (z=1) but below split/ROI (z=5–13).
        self._highlight_overlay_item.setZValue(4)
        self._scene.addItem(self._highlight_overlay_item)

    def set_overlay_alpha(self, alpha: int) -> None:
        self._alpha = max(0, min(255, alpha))
        self._refresh_overlay()

    # ----------------------------- before/after split -----------------------

    def set_split_visible(self, visible: bool) -> None:
        """Show or hide the vertical before/after divider on the canvas."""
        self._split_visible = bool(visible)
        if not visible:
            self._clear_split_items()
            if self._overlay_item is not None:
                self._overlay_item.set_split_x(None)
        elif self._split_x is not None:
            self._draw_split()
            if self._overlay_item is not None:
                self._overlay_item.set_split_x(self._split_x)

    def is_split_visible(self) -> bool:
        return self._split_visible

    def set_split_x(self, x: float) -> None:
        self._split_x = float(x)
        if self._split_visible:
            self._draw_split()
            if self._overlay_item is not None:
                self._overlay_item.set_split_x(self._split_x)
            self.split_changed.emit()

    def split_x(self) -> float | None:
        return self._split_x

    def _clear_split_items(self) -> None:
        for item in (
            self._split_line_shadow_item,
            self._split_line_item,
            self._split_handle_shadow_item,
            self._split_handle_item,
            self._split_handle_inner,
            self._split_handle_ring_item,
            self._split_arrow_item,
        ):
            if item is not None and item.scene() is self._scene:
                self._scene.removeItem(item)
        self._split_line_shadow_item = None
        self._split_line_item = None
        self._split_handle_shadow_item = None
        self._split_handle_item = None
        self._split_handle_inner = None
        self._split_handle_ring_item = None
        self._split_arrow_item = None

    def _draw_split(self) -> None:
        """Render the before/after divider line + draggable handle.

        Composition (bottom → top):
          1. Soft dark line behind the white one — gives the divider
             definition on bright dish photos.
          2. Crisp white line on top.
          3. Soft blurred drop-shadow ellipse under the handle.
          4. Solid white handle (clean canvas).
          5. Thin amber accent ring — brand colour without taking over.
          6. Dark chevrons inside, signalling 'draggable'.

        Everything in the handle uses ItemIgnoresTransformations so the
        slider stays at a constant viewport size regardless of zoom.
        """
        if self._base_item is None or self._split_x is None:
            return
        h = self._base_item.pixmap().height()
        sx = self._split_x

        # ---------------------------------------------------------- 1) line
        # Shadow line behind: subtle dark stripe so the white divider
        # stays visible over the lightest agar areas without looking neon.
        if self._split_line_shadow_item is None:
            self._split_line_shadow_item = QGraphicsLineItem()
            self._split_line_shadow_item.setZValue(9)
            self._scene.addItem(self._split_line_shadow_item)
        shadow_pen = QPen(QColor(0, 0, 0, 110), 0)
        shadow_pen.setCosmetic(True)
        shadow_pen.setWidthF(3.5)
        shadow_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self._split_line_shadow_item.setPen(shadow_pen)
        self._split_line_shadow_item.setLine(sx, 0, sx, h)

        # Main white line — slightly thinner & softer than before.
        if self._split_line_item is None:
            self._split_line_item = QGraphicsLineItem()
            self._split_line_item.setZValue(10)
            self._scene.addItem(self._split_line_item)
        line_pen = QPen(QColor(255, 255, 255, 235), 0)
        line_pen.setCosmetic(True)
        line_pen.setWidthF(1.5)
        line_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self._split_line_item.setPen(line_pen)
        self._split_line_item.setLine(sx, 0, sx, h)

        # -------------------------------------------------------- 2) handle
        r = 22.0                          # was 28 — smaller & more refined
        handle_pos = QPointF(sx, h / 2)

        # Soft drop shadow — a slightly larger dark ellipse offset down,
        # blurred via QGraphicsBlurEffect. Works with IgnoresTransforms.
        if self._split_handle_shadow_item is None:
            self._split_handle_shadow_item = QGraphicsEllipseItem()
            self._split_handle_shadow_item.setZValue(10)
            self._split_handle_shadow_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
            )
            self._scene.addItem(self._split_handle_shadow_item)
            blur = QGraphicsBlurEffect()
            blur.setBlurRadius(10)
            self._split_handle_shadow_item.setGraphicsEffect(blur)
        self._split_handle_shadow_item.setPen(QPen(Qt.PenStyle.NoPen))
        self._split_handle_shadow_item.setBrush(QBrush(QColor(0, 0, 0, 130)))
        sh_r = r + 1.5
        self._split_handle_shadow_item.setRect(QRectF(
            -sh_r, -sh_r + 3.0, sh_r * 2, sh_r * 2,
        ))
        self._split_handle_shadow_item.setPos(handle_pos)

        # Main white handle — clean, no fill colour.
        if self._split_handle_item is None:
            self._split_handle_item = QGraphicsEllipseItem()
            self._split_handle_item.setZValue(11)
            self._split_handle_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
            )
            self._scene.addItem(self._split_handle_item)
        self._split_handle_item.setPen(QPen(Qt.PenStyle.NoPen))
        self._split_handle_item.setBrush(QBrush(QColor(255, 255, 255, 250)))
        self._split_handle_item.setRect(QRectF(-r, -r, r * 2, r * 2))
        self._split_handle_item.setPos(handle_pos)

        # Thin amber accent ring on top of the white handle — keeps the
        # brand colour without making the whole control look like a button.
        if self._split_handle_ring_item is None:
            self._split_handle_ring_item = QGraphicsEllipseItem()
            self._split_handle_ring_item.setZValue(12)
            self._split_handle_ring_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
            )
            self._scene.addItem(self._split_handle_ring_item)
        ring_pen = QPen(QColor(240, 200, 74, 230), 0)
        ring_pen.setCosmetic(True)
        ring_pen.setWidthF(1.6)
        self._split_handle_ring_item.setPen(ring_pen)
        self._split_handle_ring_item.setBrush(QBrush(Qt.GlobalColor.transparent))
        # Inset by 1.5 px so the ring sits just inside the handle edge.
        self._split_handle_ring_item.setRect(QRectF(
            -r + 1.5, -r + 1.5, (r - 1.5) * 2, (r - 1.5) * 2,
        ))
        self._split_handle_ring_item.setPos(handle_pos)

        # Chevrons ‹  › — slimmer & darker for clean contrast on white.
        from PySide6.QtGui import QPainterPath
        from PySide6.QtWidgets import QGraphicsPathItem

        if self._split_arrow_item is None:
            self._split_arrow_item = QGraphicsPathItem()
            self._split_arrow_item.setZValue(13)
            self._split_arrow_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
            )
            self._scene.addItem(self._split_arrow_item)
        path = QPainterPath()
        path.moveTo(-5.0, -5.0)
        path.lineTo(-9.0, 0.0)
        path.lineTo(-5.0, 5.0)
        path.moveTo(5.0, -5.0)
        path.lineTo(9.0, 0.0)
        path.lineTo(5.0, 5.0)
        arrow_pen = QPen(QColor(35, 45, 65, 240), 0)
        arrow_pen.setCosmetic(True)
        arrow_pen.setWidthF(1.8)
        arrow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        arrow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self._split_arrow_item.setPen(arrow_pen)
        self._split_arrow_item.setPath(path)
        self._split_arrow_item.setPos(handle_pos)

    def _split_handle_hit(self, scene_pos: QPointF) -> bool:
        if not self._split_visible or self._split_x is None or self._base_item is None:
            return False
        # Convert the handle's centre back to viewport coords, compare in pixels
        handle_scene = QPointF(self._split_x, self._base_item.pixmap().height() / 2)
        handle_view = self.mapFromScene(handle_scene)
        click_view = self.mapFromScene(scene_pos)
        dx = click_view.x() - handle_view.x()
        dy = click_view.y() - handle_view.y()
        return (dx * dx + dy * dy) ** 0.5 <= self.SPLIT_HIT_RADIUS_VIEW + 4

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
            self._scene.addItem(self._overlay_item)
        else:
            self._overlay_item.setPixmap(pix)
        # If the before/after split is visible, the overlay paints only
        # right of the divider. Otherwise it covers the whole image.
        self._overlay_item.set_split_x(self._split_x if self._split_visible else None)

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

        # Split handle has highest priority — it's a bright on-canvas
        # affordance and the user expects clicking it to drag.
        if event.button() == Qt.MouseButton.LeftButton and not self._space_held:
            scene_pos = self.mapToScene(event.position().toPoint())
            if self._split_handle_hit(scene_pos):
                self._split_dragging = True
                self.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
                event.accept()
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

        # Plain left-click on an existing mask: select that colony
        # (no edit mode, no drag mode). Fires colony_clicked so the UI
        # can scroll the corresponding table row into view.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._space_held
            and self._masks is not None
            and self._edit_mode == self.EDIT_NONE
        ):
            scene_pos = self.mapToScene(event.position().toPoint())
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            h, w = self._masks.shape
            if 0 <= x < w and 0 <= y < h:
                label_id = int(self._masks[y, x])
                if label_id > 0:
                    self.set_highlighted_label(label_id)
                    self.colony_clicked.emit(label_id)
                    event.accept()
                    return
                else:
                    # Click on empty area clears the highlight
                    if self._highlight_label is not None:
                        self.set_highlighted_label(None)
                        self.colony_clicked.emit(0)
                        event.accept()
                        return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        # Split-handle drag has highest priority while active.
        if self._split_dragging and self._base_item is not None:
            scene_pos = self.mapToScene(event.position().toPoint())
            pix_w = self._base_item.pixmap().width()
            new_x = max(0.0, min(float(pix_w), float(scene_pos.x())))
            self._split_x = new_x
            self._draw_split()
            if self._overlay_item is not None:
                self._overlay_item.set_split_x(self._split_x)
            self.split_changed.emit()
            event.accept()
            return

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

        # Hover cursors: split > ROI > edit-mode > default
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            scene_pos = self.mapToScene(event.position().toPoint())
            if self._split_visible and self._split_handle_hit(scene_pos):
                self.viewport().setCursor(Qt.CursorShape.SizeHorCursor)
            elif self._roi_visible and self._roi_drag_mode == self.ROI_NONE:
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
        if self._split_dragging:
            self._split_dragging = False
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
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
