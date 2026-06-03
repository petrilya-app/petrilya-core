"""Main application window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from petrilya.export.csv_writer import write_csv
from petrilya.export.json_manifest import build_manifest, write_manifest
from petrilya.export.pdf_report import write_pdf_report
from petrilya.ui.image_view import ImageView
from petrilya.ui.worker import AnalysisWorker, BatchWorker


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Petrilya — colony counter (preview)")
        self.resize(1320, 820)
        self.setAcceptDrops(True)

        self.threadpool = QThreadPool.globalInstance()

        # state for the currently loaded image
        self.current_image_path: Path | None = None
        self.current_image: np.ndarray | None = None
        self.current_masks: np.ndarray | None = None
        self.current_metrics: list[dict] = []
        self.last_elapsed: float = 0.0
        self.last_engine_name: str = ""
        self.last_engine_params: dict = {}

        self._build_menus()
        self._build_central()
        self._build_status_bar()

    # ----------------------------- UI scaffolding ---------------------------

    def _build_menus(self) -> None:
        bar = self.menuBar()
        file_menu = bar.addMenu("&File")

        open_act = QAction("&Open image...", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self.open_dialog)
        file_menu.addAction(open_act)

        batch_act = QAction("&Batch process folder...", self)
        batch_act.setShortcut(QKeySequence("Ctrl+B"))
        batch_act.triggered.connect(self.run_batch)
        file_menu.addAction(batch_act)

        file_menu.addSeparator()

        export_csv_act = QAction("Export &CSV...", self)
        export_csv_act.setShortcut(QKeySequence("Ctrl+E"))
        export_csv_act.triggered.connect(self.export_csv)
        file_menu.addAction(export_csv_act)

        export_pdf_act = QAction("Export &PDF report...", self)
        export_pdf_act.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_pdf_act.triggered.connect(self.export_pdf)
        file_menu.addAction(export_pdf_act)

        export_json_act = QAction("Export &JSON manifest...", self)
        export_json_act.triggered.connect(self.export_json)
        file_menu.addAction(export_json_act)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    def _build_central(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # left: image
        self.image_view = ImageView()
        splitter.addWidget(self.image_view)

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

        self.gpu_check = QCheckBox("Use GPU (CUDA)")
        self.gpu_check.setChecked(False)
        form.addRow(self.gpu_check)

        right_layout.addWidget(settings_box)

        # ---- analyze button ----
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.clicked.connect(self.run_analysis)
        right_layout.addWidget(self.analyze_btn)

        # ---- view options ----
        self.overlay_check = QCheckBox("Show mask overlay")
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(self.image_view.toggle_overlay)
        right_layout.addWidget(self.overlay_check)

        # ---- summary ----
        self.summary_label = QLabel("No image loaded.")
        self.summary_label.setStyleSheet(
            "padding:8px; background:#2a2a2a; border-radius:4px;"
        )
        self.summary_label.setWordWrap(True)
        right_layout.addWidget(self.summary_label)

        # ---- per-colony table ----
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["#", "Area", "Diameter", "Eccentricity"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.table, stretch=1)

        # ---- export buttons row ----
        exports = QHBoxLayout()
        self.csv_btn = QPushButton("CSV")
        self.csv_btn.setEnabled(False)
        self.csv_btn.clicked.connect(self.export_csv)
        exports.addWidget(self.csv_btn)

        self.pdf_btn = QPushButton("PDF")
        self.pdf_btn.setEnabled(False)
        self.pdf_btn.clicked.connect(self.export_pdf)
        exports.addWidget(self.pdf_btn)

        self.json_btn = QPushButton("JSON")
        self.json_btn.setEnabled(False)
        self.json_btn.clicked.connect(self.export_json)
        exports.addWidget(self.json_btn)

        right_layout.addLayout(exports)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.setCentralWidget(splitter)

    def _build_status_bar(self) -> None:
        bar = QStatusBar()
        self.setStatusBar(bar)
        self.status_label = QLabel("Ready. Mock engine — Cellpose weights pending.")
        bar.addWidget(self.status_label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        bar.addPermanentWidget(self.progress)

    # -------------------------------- helpers -------------------------------

    def _scale_value(self) -> float | None:
        v = self.scale_spin.value()
        return v if v > 0 else None

    def _enable_exports(self, enabled: bool) -> None:
        self.csv_btn.setEnabled(enabled)
        self.pdf_btn.setEnabled(enabled)
        self.json_btn.setEnabled(enabled)

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

            arr = np.array(Image.open(path).convert("L"))
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Open failed", str(e))
            return

        self.current_image_path = path
        self.current_image = arr
        self.current_masks = None
        self.current_metrics = []
        self.image_view.set_image(arr)
        self.summary_label.setText(
            f"Loaded: {path.name}  ({arr.shape[1]}x{arr.shape[0]})"
        )
        self.table.setRowCount(0)
        self.analyze_btn.setEnabled(True)
        self._enable_exports(False)
        self.status_label.setText(f"Loaded {path.name}")

    def run_analysis(self) -> None:
        if not self.current_image_path:
            return
        self.analyze_btn.setEnabled(False)
        self._enable_exports(False)
        self.progress.setVisible(True)
        self.status_label.setText("Working...")

        worker = AnalysisWorker(
            self.current_image_path,
            use_gpu=self.gpu_check.isChecked(),
            scale_um_per_px=self._scale_value(),
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
        self.current_image = image
        self.current_masks = masks
        self.current_metrics = metrics
        self.last_elapsed = elapsed
        self.last_engine_name = engine_name
        self.last_engine_params = engine_params

        self.image_view.set_overlay(masks)

        unit = "um^2" if self._scale_value() else "px"
        self.summary_label.setText(
            f"<b>{len(metrics)}</b> colonies found in <b>{elapsed:.2f}s</b><br>"
            f"<span style='color:#aaa'>engine: {engine_name} | unit: {unit}</span>"
        )
        self.status_label.setText(f"Done — {len(metrics)} colonies in {elapsed:.2f}s")
        self._populate_table(metrics)
        self._enable_exports(bool(metrics))

    def on_error(self, msg: str) -> None:
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis failed", msg)
        self.status_label.setText("Error.")

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
            QMessageBox.information(self, "No data", "Run analysis first.")
            return
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", str(self._default_path(".csv")), "CSV (*.csv)"
        )
        if path_str:
            write_csv(self.current_metrics, Path(path_str))
            self.status_label.setText(f"Saved {path_str}")

    def export_pdf(self) -> None:
        if (
            not self.current_metrics
            or self.current_image is None
            or self.current_masks is None
        ):
            QMessageBox.information(self, "No data", "Run analysis first.")
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
                masks=self.current_masks,
                metrics=self.current_metrics,
                image_name=self.current_image_path.name
                if self.current_image_path
                else "image",
                elapsed_seconds=self.last_elapsed,
                engine_name=self.last_engine_name,
                scale_um_per_px=self._scale_value(),
            )
            self.status_label.setText(f"Saved {path_str}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "PDF export failed", str(e))

    def export_json(self) -> None:
        if not self.current_metrics or self.current_image_path is None:
            QMessageBox.information(self, "No data", "Run analysis first.")
            return
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
                masks_shape=self.current_masks.shape
                if self.current_masks is not None
                else (0, 0),
                n_objects=len(self.current_metrics),
                elapsed_seconds=self.last_elapsed,
                engine_name=self.last_engine_name,
                engine_params=self.last_engine_params,
                scale_um_per_px=self._scale_value(),
            )
            write_manifest(manifest, Path(path_str))
            self.status_label.setText(f"Saved {path_str}")
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "JSON export failed", str(e))

    # ------------------------------ batch -----------------------------------

    def run_batch(self) -> None:
        in_dir = QFileDialog.getExistingDirectory(self, "Select input folder")
        if not in_dir:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not out_dir:
            return

        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.status_label.setText("Batch: scanning...")

        worker = BatchWorker(
            Path(in_dir),
            Path(out_dir),
            use_gpu=self.gpu_check.isChecked(),
            scale_um_per_px=self._scale_value(),
        )
        worker.signals.progress.connect(self.on_batch_progress)
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
        QMessageBox.information(
            self,
            "Batch complete",
            f"Processed {ok} ok, {fail} failed.\n\nSummary: {summary_csv}",
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
