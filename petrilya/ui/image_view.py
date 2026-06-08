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
    QFrame,
    QGraphicsBlurEffect,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
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
    EDIT_LASSO = "lasso"

    # ROI dragging modes
    ROI_NONE = "none"
    ROI_MOVE = "move"
    ROI_RESIZE = "resize"
    ROI_HIT_RADIUS_SCENE = 18  # in scene units; pickable handle size

    # Split-handle dragging
    SPLIT_HIT_RADIUS_VIEW = 28  # in viewport pixels — handle is a fixed size

    masks_edited = Signal()    # emitted after every user-triggered mask change
    open_requested = Signal()  # canvas clicked while empty → open file dialog
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

        # Lasso freehand-select state
        self._lasso_points: list[QPointF] = []
        self._lasso_item = None      # QGraphicsPathItem
        self._lasso_active: bool = False

        # ROI dim-mask: a translucent dark overlay covering everything
        # OUTSIDE the dish circle so the user only "sees" their crop.
        self._roi_mask_item = None   # QGraphicsPathItem

        # Brush/erase cursor preview — a hollow circle that follows the
        # mouse showing the actual paint radius.
        self._cursor_preview_item: QGraphicsEllipseItem | None = None

        # Table-row → canvas selection: highlight one colony in vivid
        # contrast over the regular overlay.
        self._highlight_label: int | None = None
        self._highlight_overlay_item: QGraphicsPixmapItem | None = None

        # Editorial empty-state — dashed-border card with upload icon
        # and friendly two-line copy. Lives as a child widget of the
        # viewport, not a scene item, so it can use the global QSS.
        self._empty_state = self._build_empty_state()
        self._empty_state.show()
        self._placeholder = None

    # ---------------------------- public api --------------------------------

    def _build_empty_state(self) -> QFrame:
        """Build the dashed-card 'drop image here' overlay."""
        from petrilya.ui.icons import icon

        frame = QFrame(self.viewport())
        frame.setObjectName("emptyState")
        frame.setFixedSize(380, 240)
        frame.setCursor(Qt.CursorShape.PointingHandCursor)
        # Click anywhere on the dashed card to open a file picker.
        frame.mousePressEvent = (
            lambda e: self.open_requested.emit()
            if e.button() == Qt.MouseButton.LeftButton else None
        )

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(28, 32, 28, 28)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        ic = QLabel()
        ic.setPixmap(icon("upload-cloud", 42, "#8a93a4").pixmap(42, 42))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ic)

        title = QLabel("Drop a petri dish photo here")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        sub = QLabel("or use File → Open  ·  JPG, PNG, TIFF, BMP")
        sub.setObjectName("emptySubtitle")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(sub)

        layout.addSpacing(6)

        hint = QLabel("CTRL+O")
        hint.setObjectName("emptyHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(hint)

        return frame

    def _position_empty_state(self) -> None:
        if getattr(self, "_empty_state", None) is None:
            return
        vp = self.viewport()
        if vp is None:
            return
        es = self._empty_state
        x = (vp.width() - es.width()) // 2
        y = (vp.height() - es.height()) // 2
        es.move(max(0, x), max(0, y))

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt API
        super().resizeEvent(event)
        self._position_empty_state()

    def set_image(self, image: np.ndarray) -> None:
        self._scene.clear()
        self._placeholder = None
        if getattr(self, "_empty_state", None) is not None:
            self._empty_state.hide()
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
        """Render the before/after divider line + slim pill handle.

        Modern slim composition:
          1. 1-px white divider line, no shadow stripe — clean.
          2. Vertical capsule handle (10×26 px), white with subtle
             1-px border. Two tiny chevrons inside.
        Everything in the handle uses ItemIgnoresTransformations so the
        slider stays at a constant viewport size regardless of zoom.
        """
        if self._base_item is None or self._split_x is None:
            return
        h = self._base_item.pixmap().height()
        sx = self._split_x

        # 1) Crisp 1-px white divider — no shadow, no glow.
        if self._split_line_shadow_item is not None:
            if self._split_line_shadow_item.scene() is self._scene:
                self._scene.removeItem(self._split_line_shadow_item)
            self._split_line_shadow_item = None
        if self._split_line_item is None:
            self._split_line_item = QGraphicsLineItem()
            self._split_line_item.setZValue(10)
            self._scene.addItem(self._split_line_item)
        line_pen = QPen(QColor(255, 255, 255, 215), 0)
        line_pen.setCosmetic(True)
        line_pen.setWidthF(1.0)
        line_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self._split_line_item.setPen(line_pen)
        self._split_line_item.setLine(sx, 0, sx, h)

        # Discard the old big circle handle if it's still on the scene.
        for attr in ("_split_handle_shadow_item",
                     "_split_handle_item",
                     "_split_handle_ring_item",
                     "_split_handle_inner"):
            item = getattr(self, attr, None)
            if item is not None and item.scene() is self._scene:
                self._scene.removeItem(item)
            setattr(self, attr, None)

        # 2) Slim vertical pill handle as a single QGraphicsPathItem so we
        # get rounded corners "for free" via a rounded rect path.
        from PySide6.QtGui import QPainterPath
        from PySide6.QtWidgets import QGraphicsPathItem

        if self._split_arrow_item is None:
            self._split_arrow_item = QGraphicsPathItem()
            self._split_arrow_item.setZValue(13)
            self._split_arrow_item.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations
            )
            self._scene.addItem(self._split_arrow_item)

        pw, ph = 12.0, 30.0      # pill width × height in viewport pixels
        pill = QPainterPath()
        pill.addRoundedRect(-pw / 2, -ph / 2, pw, ph, pw / 2, pw / 2)
        # tiny chevrons inside, ASCII < and >
        pill.moveTo(-2.2, -3.0)
        pill.lineTo(-4.0, 0.0)
        pill.lineTo(-2.2, 3.0)
        pill.moveTo(2.2, -3.0)
        pill.lineTo(4.0, 0.0)
        pill.lineTo(2.2, 3.0)
        self._split_arrow_item.setPath(pill)
        self._split_arrow_item.setBrush(QBrush(QColor(255, 255, 255, 245)))
        outline = QPen(QColor(15, 22, 32, 200), 0)
        outline.setCosmetic(True)
        outline.setWidthF(1.0)
        outline.setCapStyle(Qt.PenCapStyle.RoundCap)
        outline.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self._split_arrow_item.setPen(outline)
        handle_pos = QPointF(sx, h / 2)
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
            self._clear_roi_mask()
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        elif self._roi is not None:
            self._draw_roi()
            self._draw_roi_mask()

    def is_roi_visible(self) -> bool:
        return self._roi_visible

    def set_dish_roi(self, cx: float, cy: float, r: float) -> None:
        self._roi = (float(cx), float(cy), max(10.0, float(r)))
        if self._roi_visible:
            self._draw_roi()
            self._draw_roi_mask()

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

    # ----------------------------- ROI dim mask ------------------------
    def _draw_roi_mask(self) -> None:
        """Crop the view to inside the ROI circle — everything outside is
        fully blacked out, as if the rest of the image isn't there.

        Implemented as a single QGraphicsPathItem with the path = canvas
        rect minus the circle (an "even-odd" fill rule cuts a hole).
        Solid opaque fill so the user sees a true crop, not a dim overlay.
        """
        from PySide6.QtGui import QPainterPath
        from PySide6.QtWidgets import QGraphicsPathItem

        if self._base_item is None or self._roi is None:
            return
        cx, cy, r = self._roi
        h = self._base_item.pixmap().height()
        w = self._base_item.pixmap().width()

        outer = QPainterPath()
        outer.addRect(0, 0, w, h)
        outer.addEllipse(cx - r, cy - r, r * 2, r * 2)
        outer.setFillRule(Qt.FillRule.OddEvenFill)

        if self._roi_mask_item is None:
            self._roi_mask_item = QGraphicsPathItem()
            # Sit above the highlight overlay (z=4) and mask overlay (z=1)
            # so colonies/text outside the circle vanish too — not just the
            # background photo.
            self._roi_mask_item.setZValue(8)
            self._scene.addItem(self._roi_mask_item)
        self._roi_mask_item.setPath(outer)
        self._roi_mask_item.setPen(QPen(Qt.PenStyle.NoPen))
        self._roi_mask_item.setBrush(QBrush(QColor(0, 0, 0, 255)))

    def _clear_roi_mask(self) -> None:
        if self._roi_mask_item is not None and self._roi_mask_item.scene() is self._scene:
            self._scene.removeItem(self._roi_mask_item)
        self._roi_mask_item = None

    def set_edit_mode(self, mode: str) -> None:
        if mode not in (self.EDIT_NONE, self.EDIT_ERASE,
                        self.EDIT_BRUSH, self.EDIT_LASSO):
            raise ValueError(f"Unknown edit mode: {mode}")
        self._edit_mode = mode
        # When the user leaves brush/erase mode the cursor-preview circle
        # should go away — it makes no sense in View / Lasso modes.
        if mode not in (self.EDIT_BRUSH, self.EDIT_ERASE):
            self._hide_cursor_preview()
        if mode == self.EDIT_NONE:
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        elif mode == self.EDIT_LASSO:
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
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
        # No image loaded → any left-click opens the file picker.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._base_item is None
        ):
            self.open_requested.emit()
            return
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

        # Lasso freehand start — only when masks exist and lasso mode is on.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._space_held
            and self._masks is not None
            and self._edit_mode == self.EDIT_LASSO
        ):
            self._start_lasso(self.mapToScene(event.position().toPoint()))
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
            self._draw_roi_mask()
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

        # Lasso freehand grow
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self._lasso_active
            and self._edit_mode == self.EDIT_LASSO
        ):
            self._extend_lasso(self.mapToScene(event.position().toPoint()))
            event.accept()
            return

        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and not self._space_held
            and self._masks is not None
            and self._edit_mode == self.EDIT_BRUSH
        ):
            self._apply_edit_at(
                self.mapToScene(event.position().toPoint()), starting=False
            )
            # also follow with the cursor preview circle
            self._update_cursor_preview(self.mapToScene(event.position().toPoint()))
            event.accept()
            return

        # Brush/erase: show a cursor-preview circle that tracks the mouse
        # at the configured radius, so the user sees the actual paint area.
        if (
            self._edit_mode in (self.EDIT_BRUSH, self.EDIT_ERASE)
            and self._masks is not None
        ):
            self._update_cursor_preview(self.mapToScene(event.position().toPoint()))

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
            # Finish a lasso path → erase every colony inside it.
            if self._lasso_active:
                self._finish_lasso()
                event.accept()
                return
            self._brush_active_label = 0
            self.masks_edited.emit()
        super().mouseReleaseEvent(event)

    # ---------------------------- lasso + cursor preview --------------------

    def _start_lasso(self, scene_pos: QPointF) -> None:
        from PySide6.QtGui import QPainterPath
        from PySide6.QtWidgets import QGraphicsPathItem

        self._lasso_points = [scene_pos]
        self._lasso_active = True
        if self._lasso_item is None:
            self._lasso_item = QGraphicsPathItem()
            self._lasso_item.setZValue(14)
            self._scene.addItem(self._lasso_item)
        path = QPainterPath(scene_pos)
        self._lasso_item.setPath(path)
        pen = QPen(QColor(91, 159, 209, 230), 0)  # accent blue
        pen.setCosmetic(True)
        pen.setWidthF(1.8)
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setDashPattern([5, 4])
        self._lasso_item.setPen(pen)
        self._lasso_item.setBrush(QBrush(QColor(91, 159, 209, 45)))

    def _extend_lasso(self, scene_pos: QPointF) -> None:
        from PySide6.QtGui import QPainterPath
        if not self._lasso_active or self._lasso_item is None:
            return
        # Throttle: skip points too close to the last one
        last = self._lasso_points[-1] if self._lasso_points else None
        if last is not None:
            dx = scene_pos.x() - last.x()
            dy = scene_pos.y() - last.y()
            if dx * dx + dy * dy < 4.0:    # < 2 scene px
                return
        self._lasso_points.append(scene_pos)
        path = QPainterPath(self._lasso_points[0])
        for p in self._lasso_points[1:]:
            path.lineTo(p)
        self._lasso_item.setPath(path)

    def _finish_lasso(self) -> None:
        from PySide6.QtGui import QPainterPath
        if not self._lasso_active or self._masks is None:
            self._lasso_active = False
            return
        pts = self._lasso_points
        self._lasso_active = False

        # Need at least a small triangle to do anything meaningful.
        if len(pts) < 3:
            self._clear_lasso_item()
            return

        # Close the path and erase every colony whose centroid lies inside.
        path = QPainterPath(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        path.closeSubpath()

        from skimage import measure
        removed = 0
        for prop in measure.regionprops(self._masks):
            cy, cx = prop.centroid
            if path.contains(QPointF(float(cx), float(cy))):
                self._masks[self._masks == prop.label] = 0
                removed += 1
        self._clear_lasso_item()
        if removed:
            self._refresh_overlay()
            self.masks_edited.emit()

    def _clear_lasso_item(self) -> None:
        if self._lasso_item is not None and self._lasso_item.scene() is self._scene:
            self._scene.removeItem(self._lasso_item)
        self._lasso_item = None
        self._lasso_points = []

    def _update_cursor_preview(self, scene_pos: QPointF) -> None:
        """A hollow circle that tracks the cursor at the brush radius."""
        if self._cursor_preview_item is None:
            self._cursor_preview_item = QGraphicsEllipseItem()
            self._cursor_preview_item.setZValue(15)
            self._cursor_preview_item.setBrush(QBrush(Qt.GlobalColor.transparent))
            self._scene.addItem(self._cursor_preview_item)
        # Colour hints at the mode — red-ish for erase, blue for brush.
        if self._edit_mode == self.EDIT_ERASE:
            col = QColor(224, 96, 96, 230)
        else:
            col = QColor(91, 159, 209, 230)
        pen = QPen(col, 0)
        pen.setCosmetic(True)
        pen.setWidthF(1.6)
        self._cursor_preview_item.setPen(pen)
        r = float(self._brush_size)
        self._cursor_preview_item.setRect(
            scene_pos.x() - r, scene_pos.y() - r, r * 2, r * 2
        )
        self._cursor_preview_item.show()

    def _hide_cursor_preview(self) -> None:
        if self._cursor_preview_item is not None:
            self._cursor_preview_item.hide()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hide_cursor_preview()
        super().leaveEvent(event)

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
