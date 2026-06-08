"""Main application window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QSize, Qt, QThreadPool
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from petrilya.export.csv_writer import write_csv
from petrilya.export.json_manifest import build_manifest, write_manifest
from petrilya.export.pdf_report import write_pdf_report
from petrilya.metrics.colony import compute_colony_metrics
from petrilya.ui.icons import icon
from petrilya.ui.image_view import ImageCanvas
from petrilya.ui.theme import current_theme, toggle_theme, PALETTES
from petrilya.ui.toast import toast
from petrilya.ui.worker import AnalysisWorker, BatchWorker


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Petrilya — colony counter (preview)")
        self.resize(1380, 860)
        self.setAcceptDrops(True)

        self.threadpool = QThreadPool.globalInstance()

        # state for the currently loaded image
        self.current_image_path: Path | None = None
        self.current_image: np.ndarray | None = None
        self.current_metrics: list[dict] = []
        self.last_elapsed: float = 0.0
        self.last_engine_name: str = ""
        self.last_engine_params: dict = {}

        self._build_actions()
        self._build_menus()
        self._build_toolbar()
        self._build_central()
        self._build_status_bar()

        # store the right pane for sidebar collapse toggle
        # (populated in _build_central via self._right_pane)

    # ----------------------------- UI scaffolding ---------------------------

    def _build_actions(self) -> None:
        """Build QActions once — reused by both the menubar and toolbar."""
        # File ----------------------------------------------------------------
        self.act_open = QAction(icon("folder-open"), "Open image…", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.open_dialog)

        self.act_batch = QAction(icon("folder"), "Batch folder…", self)
        self.act_batch.setShortcut(QKeySequence("Ctrl+B"))
        self.act_batch.triggered.connect(self.run_batch)

        self.act_analyze = QAction(icon("zap"), "Analyze", self)
        self.act_analyze.setShortcut(QKeySequence("Ctrl+R"))
        self.act_analyze.triggered.connect(self.run_analysis)
        self.act_analyze.setEnabled(False)

        self.act_export_csv = QAction(icon("file-down"), "Export CSV…", self)
        self.act_export_csv.setShortcut(QKeySequence("Ctrl+E"))
        self.act_export_csv.triggered.connect(self.export_csv)
        self.act_export_csv.setEnabled(False)

        self.act_export_pdf = QAction(icon("file-text"), "Export PDF…", self)
        self.act_export_pdf.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.act_export_pdf.triggered.connect(self.export_pdf)
        self.act_export_pdf.setEnabled(False)

        self.act_export_json = QAction(icon("braces"), "Export JSON…", self)
        self.act_export_json.triggered.connect(self.export_json)
        self.act_export_json.setEnabled(False)

        self.act_quit = QAction("Quit", self)
        self.act_quit.setShortcut(QKeySequence.StandardKey.Quit)
        self.act_quit.triggered.connect(self.close)

        # View ----------------------------------------------------------------
        self.act_fit = QAction("Fit to window", self)
        self.act_fit.setShortcut(QKeySequence("F"))
        self.act_fit.triggered.connect(self._fit_canvas)

        self.act_reset_zoom = QAction("Reset zoom (100%)", self)
        self.act_reset_zoom.setShortcut(QKeySequence("0"))
        self.act_reset_zoom.triggered.connect(self._reset_canvas_zoom)

        # Sidebar toggle ------------------------------------------------------
        self.act_toggle_sidebar = QAction(
            icon("panel-right-close"), "Hide sidebar", self
        )
        self.act_toggle_sidebar.setShortcut(QKeySequence("Ctrl+\\"))
        self.act_toggle_sidebar.setCheckable(True)
        self.act_toggle_sidebar.toggled.connect(self._on_sidebar_toggled)

        # Theme toggle --------------------------------------------------------
        theme = current_theme()
        self.act_theme = QAction(
            icon("sun" if theme == "dark" else "moon"),
            "Switch theme",
            self,
        )
        self.act_theme.setShortcut(QKeySequence("Ctrl+T"))
        self.act_theme.triggered.connect(self._on_theme_toggle)

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")
        file_menu.addAction(self.act_open)
        file_menu.addAction(self.act_batch)
        file_menu.addSeparator()
        file_menu.addAction(self.act_export_csv)
        file_menu.addAction(self.act_export_pdf)
        file_menu.addAction(self.act_export_json)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        view_menu = bar.addMenu("&View")
        view_menu.addAction(self.act_fit)
        view_menu.addAction(self.act_reset_zoom)
        view_menu.addSeparator()
        view_menu.addAction(self.act_toggle_sidebar)
        view_menu.addAction(self.act_theme)

    def _build_toolbar(self) -> None:
        """Compact top toolbar with primary actions + chrome controls."""
        tb = QToolBar("Main", self)
        tb.setObjectName("mainToolbar")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setIconSize(QSize(17, 17))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)

        tb.addAction(self.act_open)
        tb.addAction(self.act_batch)
        tb.addSeparator()
        tb.addAction(self.act_analyze)
        tb.addSeparator()
        tb.addAction(self.act_export_csv)
        tb.addAction(self.act_export_pdf)
        tb.addAction(self.act_export_json)

        # right-aligned spacer pushes chrome controls to the far edge
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)

        tb.addAction(self.act_toggle_sidebar)
        tb.addAction(self.act_theme)
        self._toolbar = tb

    def _build_central(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # left: canvas + (hidden until needed) batch filmstrip
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self.canvas = ImageCanvas()
        self.canvas.masks_edited.connect(self.on_masks_edited)
        self.canvas.colony_clicked.connect(self._on_colony_clicked)
        self.canvas.open_requested.connect(self.open_dialog)
        left_layout.addWidget(self.canvas, 1)

        # Batch results filmstrip — populated by BatchWorker signals,
        # hidden until the first result arrives.
        self.filmstrip = self._build_filmstrip()
        self.filmstrip.setVisible(False)
        left_layout.addWidget(self.filmstrip)

        splitter.addWidget(left_pane)

        # right: controls + results
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)

        # ---- analysis settings group ----
        settings_box = QGroupBox("Analysis")
        form = QFormLayout(settings_box)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.0, 1000.0)
        self.scale_spin.setDecimals(4)
        self.scale_spin.setSingleStep(0.01)
        self.scale_spin.setValue(0.0)
        self.scale_spin.setSuffix(" um/px")
        self.scale_spin.setSpecialValueText("(pixel units)")
        form.addRow("Scale:", self.scale_spin)

        self.gpu_check = QCheckBox("Use GPU (CUDA/MPS)")
        self.gpu_check.setChecked(True)
        form.addRow(self.gpu_check)

        right_layout.addWidget(settings_box)

        # ---- analyze button (primary CTA — also mirrored in toolbar) ----
        self.analyze_btn = QPushButton("  Analyze")
        self.analyze_btn.setObjectName("primaryButton")
        self.analyze_btn.setIcon(icon("zap", 18, "#1a1304"))
        self.analyze_btn.setIconSize(QSize(18, 18))
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setMinimumHeight(44)
        self.analyze_btn.clicked.connect(self.run_analysis)
        right_layout.addWidget(self.analyze_btn)

        # ---- view & edit group ----
        view_box = QGroupBox("View & edit")
        view_layout = QVBoxLayout(view_box)

        # transparency slider
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Overlay:"))
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setRange(0, 100)
        self.alpha_slider.setValue(43)  # ~110/255
        self.alpha_slider.valueChanged.connect(self._on_alpha_changed)
        self.alpha_value_label = QLabel("43%")
        self.alpha_value_label.setMinimumWidth(36)
        alpha_row.addWidget(self.alpha_slider)
        alpha_row.addWidget(self.alpha_value_label)
        view_layout.addLayout(alpha_row)

        # before/after split toggle — drag the handle ON the image
        split_row = QHBoxLayout()
        self.split_check = QCheckBox("Compare before / after")
        self.split_check.setToolTip(
            "Show a draggable vertical divider on the image. Left of the "
            "line = original photo; right of the line = result with masks."
        )
        self.split_check.toggled.connect(self._on_split_toggled)
        split_row.addWidget(self.split_check)
        view_layout.addLayout(split_row)

        # manual dish ROI
        roi_row = QHBoxLayout()
        self.roi_check = QCheckBox("Set dish boundary manually")
        self.roi_check.setToolTip(
            "Drag the centre handle to move the circle, drag the right-edge "
            "handle to resize it. The engine will look for colonies only "
            "inside this circle and ignore the rim and background."
        )
        self.roi_check.toggled.connect(self._on_roi_toggled)
        roi_row.addWidget(self.roi_check)
        view_layout.addLayout(roi_row)

        # edit-mode toggle row — icon buttons with tooltips
        edit_row = QHBoxLayout()
        edit_row.setSpacing(6)
        edit_row.addWidget(QLabel("Mode:"))

        def _mk_mode_btn(label: str, icon_name: str, tooltip: str) -> QToolButton:
            b = QToolButton()
            b.setText(label)
            b.setIcon(icon(icon_name, 16))
            b.setIconSize(QSize(16, 16))
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            b.setCheckable(True)
            b.setAutoExclusive(True)
            b.setToolTip(tooltip)
            b.setProperty("modeButton", True)
            return b

        self.mode_view = _mk_mode_btn("View", "eye",
            "View only — pan and zoom, no editing")
        self.mode_erase = _mk_mode_btn("Erase", "eraser",
            "Click a colony to delete it from the mask")
        self.mode_brush = _mk_mode_btn("Brush", "brush",
            "Paint a new colony. Use the Brush slider to size it; "
            "scroll wheel adjusts size on the fly.")
        self.mode_lasso = _mk_mode_btn("Lasso", "lasso",
            "Drag to draw a freehand region — releasing erases every "
            "colony whose centroid is inside it.")

        self.mode_view.setChecked(True)
        for b in (self.mode_view, self.mode_erase, self.mode_brush, self.mode_lasso):
            b.toggled.connect(self._on_mode_changed)
            edit_row.addWidget(b)
        edit_row.addStretch(1)
        view_layout.addLayout(edit_row)

        # brush size
        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush:"))
        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(2, 60)
        self.brush_slider.setValue(14)
        self.brush_slider.valueChanged.connect(self.canvas.set_brush_size)
        self.brush_slider.valueChanged.connect(
            lambda v: self.brush_value_label.setText(f"{v}px")
        )
        self.brush_value_label = QLabel("14px")
        self.brush_value_label.setMinimumWidth(36)
        brush_row.addWidget(self.brush_slider)
        brush_row.addWidget(self.brush_value_label)
        view_layout.addLayout(brush_row)

        hint = QLabel(
            "WHEEL · ZOOM   ·   SPACE-DRAG · PAN   ·   F · FIT   ·   0 · RESET"
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        view_layout.addWidget(hint)

        right_layout.addWidget(view_box)

        # ---- summary ----
        self.summary_label = QLabel("No image loaded.")
        self.summary_label.setObjectName("summaryLabel")
        self.summary_label.setWordWrap(True)
        right_layout.addWidget(self.summary_label)

        # ---- per-colony table ----
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["#", "Area", "Diameter", "Eccentricity"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        # Single-row selection: pick a colony by row → highlight on canvas
        from PySide6.QtWidgets import QAbstractItemView
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        right_layout.addWidget(self.table, stretch=1)

        # ---- export buttons row ----
        exports = QHBoxLayout()
        exports.setSpacing(8)
        self.csv_btn = QPushButton("  CSV")
        self.csv_btn.setObjectName("ghostButton")
        self.csv_btn.setIcon(icon("file-down", 16, "#c4ccdb"))
        self.csv_btn.setIconSize(QSize(16, 16))
        self.csv_btn.setEnabled(False)
        self.csv_btn.clicked.connect(self.export_csv)
        exports.addWidget(self.csv_btn)

        self.pdf_btn = QPushButton("  PDF")
        self.pdf_btn.setObjectName("ghostButton")
        self.pdf_btn.setIcon(icon("file-text", 16, "#c4ccdb"))
        self.pdf_btn.setIconSize(QSize(16, 16))
        self.pdf_btn.setEnabled(False)
        self.pdf_btn.clicked.connect(self.export_pdf)
        exports.addWidget(self.pdf_btn)

        self.json_btn = QPushButton("  JSON")
        self.json_btn.setObjectName("ghostButton")
        self.json_btn.setIcon(icon("braces", 16, "#c4ccdb"))
        self.json_btn.setIconSize(QSize(16, 16))
        self.json_btn.setEnabled(False)
        self.json_btn.clicked.connect(self.export_json)
        exports.addWidget(self.json_btn)

        right_layout.addLayout(exports)

        splitter.addWidget(right)
        # Keep the canvas growing when the window resizes; sidebar stays
        # close to its preferred width unless the user drags the handle.
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        right.setMinimumWidth(420)
        right.setMaximumWidth(560)
        # Sized for a 1380 px window: ~880 canvas + 460 sidebar.
        splitter.setSizes([900, 460])

        self._splitter = splitter
        self._right_pane = right
        self._sidebar_last_sizes: list[int] | None = None

        self.setCentralWidget(splitter)

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.status_label = QLabel("Ready. Mock engine — Cellpose weights pending.")
        bar.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(240)
        self.progress.setVisible(False)
        self.progress.setTextVisible(True)
        bar.addPermanentWidget(self.progress)

    # -------------------------------- helpers -------------------------------

    def _scale_value(self) -> float | None:
        v = self.scale_spin.value()
        return v if v > 0 else None

    def _enable_exports(self, enabled: bool) -> None:
        self.csv_btn.setEnabled(enabled)
        self.pdf_btn.setEnabled(enabled)
        self.json_btn.setEnabled(enabled)
        # mirror to toolbar actions
        self.act_export_csv.setEnabled(enabled)
        self.act_export_pdf.setEnabled(enabled)
        self.act_export_json.setEnabled(enabled)

    # ----------------------------- batch filmstrip --------------------------

    def _build_filmstrip(self) -> QWidget:
        """Horizontal strip of batch-result thumbnails under the canvas.

        Populated incrementally by BatchWorker.result_ready signals.
        Clicking a thumbnail reloads that image + saved masks back into
        the canvas, with full metrics and summary — no re-segmentation.
        """
        from PySide6.QtWidgets import QListView, QListWidget

        wrap = QWidget()
        wrap_layout = QVBoxLayout(wrap)
        wrap_layout.setContentsMargins(8, 6, 8, 8)
        wrap_layout.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(4, 0, 4, 0)
        self.filmstrip_label = QLabel("BATCH RESULTS")
        self.filmstrip_label.setObjectName("hintLabel")
        header.addWidget(self.filmstrip_label)
        header.addStretch(1)
        close_btn = QToolButton()
        close_btn.setIcon(icon("x", 14))
        close_btn.setAutoRaise(True)
        close_btn.setToolTip("Hide batch filmstrip")
        close_btn.clicked.connect(lambda: self.filmstrip.setVisible(False))
        header.addWidget(close_btn)
        wrap_layout.addLayout(header)

        self.filmstrip_list = QListWidget()
        self.filmstrip_list.setObjectName("filmstrip")
        self.filmstrip_list.setViewMode(QListView.ViewMode.IconMode)
        self.filmstrip_list.setFlow(QListView.Flow.LeftToRight)
        self.filmstrip_list.setWrapping(False)
        self.filmstrip_list.setMovement(QListView.Movement.Static)
        self.filmstrip_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.filmstrip_list.setIconSize(QSize(78, 78))
        self.filmstrip_list.setGridSize(QSize(108, 122))
        self.filmstrip_list.setSpacing(6)
        self.filmstrip_list.setFixedHeight(146)
        self.filmstrip_list.setHorizontalScrollMode(
            QListView.ScrollMode.ScrollPerPixel
        )
        self.filmstrip_list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.filmstrip_list.itemClicked.connect(self._on_filmstrip_click)
        wrap_layout.addWidget(self.filmstrip_list)
        return wrap

    def _on_batch_result(
        self,
        image_path: Path,
        masks_npz: Path,
        metrics: list[dict],
        elapsed: float,
        engine_name: str,
    ) -> None:
        """Add one entry to the filmstrip as each batch image finishes."""
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QListWidgetItem

        pix = QPixmap(str(image_path))
        if pix.isNull():
            return
        # Square thumbnail with the centre cropped.
        side = min(pix.width(), pix.height())
        cx = (pix.width() - side) // 2
        cy = (pix.height() - side) // 2
        thumb = pix.copy(cx, cy, side, side).scaled(
            78, 78,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        item = QListWidgetItem(QIcon(thumb), f"{image_path.name}\n{len(metrics)} col.")
        item.setToolTip(
            f"{image_path.name}\n"
            f"{len(metrics)} colonies · {elapsed:.2f}s · {engine_name}"
        )
        item.setData(Qt.ItemDataRole.UserRole, {
            "image_path": image_path,
            "masks_npz":  masks_npz,
            "metrics":    metrics,
            "elapsed":    elapsed,
            "engine":     engine_name,
        })
        self.filmstrip_list.addItem(item)
        self.filmstrip.setVisible(True)

    def _on_filmstrip_click(self, item) -> None:
        """Load the clicked batch result back into the canvas + sidebar."""
        data = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(data, dict):
            return
        img_path: Path = data["image_path"]
        masks_npz: Path = data["masks_npz"]
        metrics: list[dict] = data["metrics"]
        elapsed: float = data["elapsed"]
        engine: str = data["engine"]

        # 1) reload the source image into the canvas
        try:
            self.load_image(img_path)
        except Exception as e:  # noqa: BLE001
            toast(self, f"Couldn't reopen {img_path.name}: {e}", level="error")
            return

        # 2) apply the saved masks
        try:
            with np.load(masks_npz) as data_np:
                masks = data_np["masks"]
            self.canvas.set_masks(masks)
        except Exception as e:  # noqa: BLE001
            toast(self, f"Masks unavailable for {img_path.name}: {e}",
                  level="error", duration_ms=5000)
            return

        # 3) replay summary + table + enable exports
        self.current_metrics = metrics
        self.last_elapsed = elapsed
        self.last_engine_name = engine
        unit = "um^2" if self._scale_value() else "px"
        self.summary_label.setText(
            f"<b>{len(metrics)}</b> colonies · "
            f"<b>{elapsed:.2f}s</b><br>"
            f"<span style='color:#8a93a4'>engine: {engine} | unit: {unit}</span>"
        )
        self._populate_table(metrics)
        self._enable_exports(bool(metrics))

    # ----------------------------- chrome handlers --------------------------

    def _on_sidebar_toggled(self, checked: bool) -> None:
        """Collapse or restore the right control panel.

        When collapsed we stash the splitter sizes so reopening returns to
        the exact previous layout — even if the user resized the splitter.
        """
        if checked:
            # collapsing
            self._sidebar_last_sizes = self._splitter.sizes()
            total = sum(self._sidebar_last_sizes)
            self._splitter.setSizes([total, 0])
            self.act_toggle_sidebar.setIcon(icon("panel-right-open"))
            self.act_toggle_sidebar.setText("Show sidebar")
        else:
            sizes = self._sidebar_last_sizes or [3, 2]
            self._splitter.setSizes(sizes)
            self.act_toggle_sidebar.setIcon(icon("panel-right-close"))
            self.act_toggle_sidebar.setText("Hide sidebar")

    def _on_theme_toggle(self) -> None:
        """Switch dark↔light, re-apply stylesheet, persist, refresh icons."""
        from petrilya.ui.app import apply_theme

        new = toggle_theme()
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, new)

        # Refresh icon colours so they read on the new theme.
        palette = PALETTES[new]
        text_strong = palette["text"]
        text_sub = palette["text_subtle"]
        accent_text = palette["accent_text"]

        self.act_open.setIcon(icon("folder-open", color=text_sub))
        self.act_batch.setIcon(icon("folder", color=text_sub))
        self.act_analyze.setIcon(icon("zap", color=text_sub))
        self.act_export_csv.setIcon(icon("file-down", color=text_sub))
        self.act_export_pdf.setIcon(icon("file-text", color=text_sub))
        self.act_export_json.setIcon(icon("braces", color=text_sub))
        self.act_toggle_sidebar.setIcon(
            icon("panel-right-open" if self.act_toggle_sidebar.isChecked()
                 else "panel-right-close",
                 color=text_sub)
        )
        self.act_theme.setIcon(
            icon("sun" if new == "dark" else "moon", color=text_sub)
        )
        # the primary CTA in the sidebar uses accent_text colour
        self.analyze_btn.setIcon(icon("zap", 18, accent_text))
        self.csv_btn.setIcon(icon("file-down", 16, text_sub))
        self.pdf_btn.setIcon(icon("file-text", 16, text_sub))
        self.json_btn.setIcon(icon("braces", 16, text_sub))

        toast(self, f"Theme: {new}", level="info", duration_ms=1800)

    def _fit_canvas(self) -> None:
        self.canvas.fit_to_window()

    def _reset_canvas_zoom(self) -> None:
        self.canvas.reset_zoom()

    def _on_alpha_changed(self, percent: int) -> None:
        self.alpha_value_label.setText(f"{percent}%")
        self.canvas.set_overlay_alpha(int(round(percent * 255 / 100)))

    def _on_split_toggled(self, checked: bool) -> None:
        self.canvas.set_split_visible(checked)

    def _on_roi_toggled(self, checked: bool) -> None:
        self.canvas.set_roi_visible(checked)

    # ---- table ↔ canvas selection link ----

    def _on_table_selection_changed(self) -> None:
        """Highlight the colony belonging to the selected table row."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.canvas.set_highlighted_label(None)
            return
        row = rows[0].row()
        if 0 <= row < len(self.current_metrics):
            label_id = int(self.current_metrics[row].get("id", 0))
            self.canvas.set_highlighted_label(label_id)
            self.status_label.setText(f"Selected colony #{label_id}")

    def _on_colony_clicked(self, label_id: int) -> None:
        """Canvas click on a colony → scroll to & select its row."""
        if not label_id:
            self.table.clearSelection()
            return
        for row, m in enumerate(self.current_metrics):
            if int(m.get("id", 0)) == int(label_id):
                # Block recursion: setting selection fires
                # _on_table_selection_changed which would set the
                # highlight again (already set), but cost is trivial
                # so we don't bother blocking signals.
                self.table.selectRow(row)
                item = self.table.item(row, 0)
                if item is not None:
                    self.table.scrollToItem(
                        item,
                        self.table.ScrollHint.PositionAtCenter,
                    )
                self.status_label.setText(f"Selected colony #{label_id}")
                break

    def _on_mode_changed(self) -> None:
        if self.mode_view.isChecked():
            self.canvas.set_edit_mode(ImageCanvas.EDIT_NONE)
            self.brush_slider.setEnabled(False)
        elif self.mode_erase.isChecked():
            self.canvas.set_edit_mode(ImageCanvas.EDIT_ERASE)
            self.brush_slider.setEnabled(False)
        elif self.mode_brush.isChecked():
            self.canvas.set_edit_mode(ImageCanvas.EDIT_BRUSH)
            self.brush_slider.setEnabled(True)
        elif self.mode_lasso.isChecked():
            self.canvas.set_edit_mode(ImageCanvas.EDIT_LASSO)
            self.brush_slider.setEnabled(False)

    # -------------------------------- actions -------------------------------

    def open_dialog(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp)",
        )
        if path_str:
            self.load_image(Path(path_str))

    def load_image(self, path: Path) -> None:
        try:
            from PIL import Image

            img = Image.open(path)
            # Display in colour; the engine workers do their own grayscale
            # conversion from the file path so we don't lose information here.
            if img.mode in ("RGB", "RGBA"):
                arr = np.array(img)
            elif img.mode == "L":
                arr = np.array(img)
            else:
                arr = np.array(img.convert("RGB"))
        except Exception as e:  # noqa: BLE001
            toast(self, f"Couldn't open image: {e}", level="error", duration_ms=6000)
            return

        self.current_image_path = path
        self.current_image = arr
        self.current_metrics = []
        self.canvas.set_image(arr)
        h, w = arr.shape[:2]
        self.summary_label.setText(
            f"Loaded: {path.name}  ({w}x{h})"
        )
        self.table.setRowCount(0)
        self.analyze_btn.setEnabled(True)
        self.act_analyze.setEnabled(True)
        self._enable_exports(False)
        self.status_label.setText(f"Loaded {path.name}")
        toast(self, f"Loaded {path.name} · {w}×{h}", level="info", duration_ms=2200)

    def run_analysis(self) -> None:
        if not self.current_image_path:
            return
        self.analyze_btn.setEnabled(False)
        self.act_analyze.setEnabled(False)
        self._enable_exports(False)
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.status_label.setText("Working...")

        # Pass the manual dish ROI if the user enabled it; otherwise let
        # the engine auto-detect via Hough.
        roi = self.canvas.dish_roi() if self.canvas.is_roi_visible() else None
        worker = AnalysisWorker(
            self.current_image_path,
            use_gpu=self.gpu_check.isChecked(),
            scale_um_per_px=self._scale_value(),
            dish_roi=roi,
        )
        worker.signals.progress.connect(self.status_label.setText)
        worker.signals.finished.connect(self.on_finished)
        worker.signals.error.connect(self.on_error)
        self.threadpool.start(worker)

    def on_finished(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        metrics: list[dict],
        elapsed: float,
        engine_name: str,
        engine_params: dict,
    ) -> None:
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.act_analyze.setEnabled(True)
        self.current_image = image
        self.current_metrics = metrics
        self.last_elapsed = elapsed
        self.last_engine_name = engine_name
        self.last_engine_params = engine_params

        self.canvas.set_masks(masks)

        unit = "um^2" if self._scale_value() else "px"
        # If the engine reported shape-filter stats, surface them — gives
        # the user honest insight into how much Cellpose's raw output
        # was trimmed (text strokes, scratches etc.).
        filter_note = ""
        stats = engine_params.get("filter_stats") if isinstance(engine_params, dict) else None
        if stats and stats.get("before", 0) > stats.get("kept", 0):
            dropped = stats["before"] - stats["kept"]
            filter_note = (
                f"<br><span style='color:#888'>"
                f"raw {stats['before']} → {stats['kept']} after shape filter "
                f"(–{dropped} non-circular)"
                f"</span>"
            )

        self.summary_label.setText(
            f"<b>{len(metrics)}</b> colonies found in <b>{elapsed:.2f}s</b><br>"
            f"<span style='color:#aaa'>engine: {engine_name} | unit: {unit}</span>"
            f"{filter_note}"
        )
        self.status_label.setText(f"Done — {len(metrics)} colonies in {elapsed:.2f}s")
        self._populate_table(metrics)
        self._enable_exports(bool(metrics))
        if metrics:
            toast(
                self,
                f"Found {len(metrics)} colonies in {elapsed:.2f}s",
                level="success",
                duration_ms=2800,
            )

    def on_error(self, msg: str) -> None:
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.act_analyze.setEnabled(True)
        # Strip the traceback for the toast — keep it short.
        first_line = msg.split("\n", 1)[0]
        toast(self, f"Analysis failed: {first_line}", level="error", duration_ms=8000)
        self.status_label.setText("Error.")

    def on_masks_edited(self) -> None:
        """Recompute metrics after the user edits masks in the canvas."""
        masks = self.canvas.masks()
        if masks is None:
            return
        self.current_metrics = compute_colony_metrics(
            masks, scale_um_per_px=self._scale_value()
        )
        unit = "um^2" if self._scale_value() else "px"
        self.summary_label.setText(
            f"<b>{len(self.current_metrics)}</b> colonies (after manual edit)<br>"
            f"<span style='color:#aaa'>engine: {self.last_engine_name} | unit: {unit}</span>"
        )
        self._populate_table(self.current_metrics)
        self._enable_exports(bool(self.current_metrics))
        self.status_label.setText(
            f"Manual edit — {len(self.current_metrics)} colonies"
        )

    def _populate_table(self, metrics: list[dict]) -> None:
        if not metrics:
            self.table.setRowCount(0)
            return

        has_um = "area_um2" in metrics[0]
        if has_um:
            self.table.setHorizontalHeaderLabels(
                ["#", "Area (um^2)", "Diameter (um)", "Eccentricity"]
            )
        else:
            self.table.setHorizontalHeaderLabels(
                ["#", "Area (px)", "Diameter (px)", "Eccentricity"]
            )

        self.table.setRowCount(len(metrics))
        for row, m in enumerate(metrics):
            self.table.setItem(row, 0, QTableWidgetItem(str(m["id"])))
            if has_um:
                self.table.setItem(row, 1, QTableWidgetItem(f"{m['area_um2']:.2f}"))
                self.table.setItem(
                    row, 2, QTableWidgetItem(f"{m['equivalent_diameter_um']:.2f}")
                )
            else:
                self.table.setItem(row, 1, QTableWidgetItem(str(m["area_px"])))
                self.table.setItem(
                    row, 2, QTableWidgetItem(f"{m['equivalent_diameter_px']:.1f}")
                )
            self.table.setItem(row, 3, QTableWidgetItem(f"{m['eccentricity']:.2f}"))

    # ------------------------------ exports ---------------------------------

    def _default_path(self, suffix: str) -> Path:
        if self.current_image_path:
            return self.current_image_path.with_suffix(suffix)
        return Path(f"colonies{suffix}")

    def export_csv(self) -> None:
        if not self.current_metrics:
            toast(self, "Run analysis first", level="info")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", str(self._default_path(".csv")), "CSV (*.csv)"
        )
        if path_str:
            write_csv(self.current_metrics, Path(path_str))
            self.status_label.setText(f"Saved {path_str}")
            toast(self, f"Saved {Path(path_str).name}", level="success")

    def export_pdf(self) -> None:
        masks = self.canvas.masks()
        if (
            not self.current_metrics
            or self.current_image is None
            or masks is None
        ):
            toast(self, "Run analysis first", level="info")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save PDF report",
            str(self._default_path(".report.pdf")),
            "PDF (*.pdf)",
        )
        if not path_str:
            return
        try:
            write_pdf_report(
                Path(path_str),
                image=self.current_image,
                masks=masks,
                metrics=self.current_metrics,
                image_name=self.current_image_path.name
                if self.current_image_path
                else "image",
                elapsed_seconds=self.last_elapsed,
                engine_name=self.last_engine_name,
                scale_um_per_px=self._scale_value(),
            )
            self.status_label.setText(f"Saved {path_str}")
            toast(self, f"Saved {Path(path_str).name}", level="success")
        except Exception as e:  # noqa: BLE001
            toast(self, f"PDF export failed: {e}", level="error", duration_ms=6000)

    def export_json(self) -> None:
        if not self.current_metrics or self.current_image_path is None:
            toast(self, "Run analysis first", level="info")
            return
        masks = self.canvas.masks()
        path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON manifest",
            str(self._default_path(".manifest.json")),
            "JSON (*.json)",
        )
        if not path_str:
            return
        try:
            manifest = build_manifest(
                image_path=self.current_image_path,
                masks_shape=masks.shape if masks is not None else (0, 0),
                n_objects=len(self.current_metrics),
                elapsed_seconds=self.last_elapsed,
                engine_name=self.last_engine_name,
                engine_params=self.last_engine_params,
                scale_um_per_px=self._scale_value(),
            )
            write_manifest(manifest, Path(path_str))
            self.status_label.setText(f"Saved {path_str}")
            toast(self, f"Saved {Path(path_str).name}", level="success")
        except Exception as e:  # noqa: BLE001
            toast(self, f"JSON export failed: {e}", level="error", duration_ms=6000)

    # ------------------------------ batch -----------------------------------

    def run_batch(self) -> None:
        in_dir = QFileDialog.getExistingDirectory(self, "Select input folder")
        if not in_dir:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir:
            return

        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Batch %v/%m (%p%)")
        self.progress.setVisible(True)
        self.status_label.setText("Batch: scanning...")

        # reset filmstrip — fresh batch, fresh thumbnails
        self.filmstrip_list.clear()
        self.filmstrip.setVisible(False)

        worker = BatchWorker(
            Path(in_dir),
            Path(out_dir),
            use_gpu=self.gpu_check.isChecked(),
            scale_um_per_px=self._scale_value(),
        )
        worker.signals.progress.connect(self.on_batch_progress)
        worker.signals.result_ready.connect(self._on_batch_result)
        worker.signals.finished.connect(self.on_batch_done)
        worker.signals.error.connect(self.on_error)
        self.threadpool.start(worker)

    def on_batch_progress(self, current: int, total: int, name: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        self.status_label.setText(f"Batch {current}/{total}: {name}")

    def on_batch_done(self, ok: int, fail: int, summary_csv: Path) -> None:
        self.progress.setVisible(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("")
        level = "success" if fail == 0 else "info"
        toast(
            self,
            f"Batch done · {ok} OK · {fail} failed → {summary_csv.name}",
            level=level,
            duration_ms=5000,
        )
        self.status_label.setText(
            f"Batch done — {ok} ok, {fail} failed. Summary: {summary_csv.name}"
        )

    # ------------------------------ drag & drop ------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 (Qt API)
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(Path(u.toLocalFile()).suffix.lower() in SUPPORTED_EXT for u in urls):
                event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 (Qt API)
        for url in event.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.suffix.lower() in SUPPORTED_EXT:
                self.load_image(p)
                return
