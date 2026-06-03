"""Main application window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
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
from petrilya.ui.image_view import ImageView
from petrilya.ui.worker import AnalysisWorker


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Petrilya — colony counter (preview)")
        self.resize(1280, 800)
        self.setAcceptDrops(True)

        self.threadpool = QThreadPool.globalInstance()
        self.current_image_path: Path | None = None
        self.current_metrics: list[dict] = []

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

        export_act = QAction("&Export CSV...", self)
        export_act.setShortcut(QKeySequence("Ctrl+E"))
        export_act.triggered.connect(self.export_csv)
        file_menu.addAction(export_act)

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

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setMinimumHeight(40)
        self.analyze_btn.clicked.connect(self.run_analysis)
        right_layout.addWidget(self.analyze_btn)

        self.gpu_check = QCheckBox("Use GPU (CUDA)")
        self.gpu_check.setChecked(False)
        right_layout.addWidget(self.gpu_check)

        self.overlay_check = QCheckBox("Show mask overlay")
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(self.image_view.toggle_overlay)
        right_layout.addWidget(self.overlay_check)

        self.summary_label = QLabel("No image loaded.")
        self.summary_label.setStyleSheet("padding:8px; background:#2a2a2a; border-radius:4px;")
        right_layout.addWidget(self.summary_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["#", "Area (px)", "Diameter (px)", "Eccentricity"])
        self.table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.table, stretch=1)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self.export_csv)
        right_layout.addWidget(self.export_btn)

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
        self.progress.setRange(0, 0)  # indeterminate when shown
        self.progress.setMaximumWidth(180)
        self.progress.setVisible(False)
        bar.addPermanentWidget(self.progress)

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
        self.current_metrics = []
        self.image_view.set_image(arr)
        self.summary_label.setText(
            f"Loaded: {path.name}  ({arr.shape[1]}x{arr.shape[0]})"
        )
        self.table.setRowCount(0)
        self.analyze_btn.setEnabled(True)
        self.export_btn.setEnabled(False)
        self.status_label.setText(f"Loaded {path.name}")

    def run_analysis(self) -> None:
        if not self.current_image_path:
            return
        self.analyze_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status_label.setText("Working...")

        worker = AnalysisWorker(self.current_image_path, use_gpu=self.gpu_check.isChecked())
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
    ) -> None:
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        self.current_metrics = metrics

        self.image_view.set_overlay(masks)
        self.summary_label.setText(
            f"<b>{len(metrics)}</b> colonies found in <b>{elapsed:.2f}s</b>  "
            f"(mock engine — replace with Cellpose once available)"
        )
        self.status_label.setText(f"Done — {len(metrics)} colonies in {elapsed:.2f}s")
        self._populate_table(metrics)
        self.export_btn.setEnabled(bool(metrics))

    def on_error(self, msg: str) -> None:
        self.progress.setVisible(False)
        self.analyze_btn.setEnabled(True)
        QMessageBox.critical(self, "Analysis failed", msg)
        self.status_label.setText("Error.")

    def _populate_table(self, metrics: list[dict]) -> None:
        self.table.setRowCount(len(metrics))
        for row, m in enumerate(metrics):
            self.table.setItem(row, 0, QTableWidgetItem(str(m["id"])))
            self.table.setItem(row, 1, QTableWidgetItem(str(m["area_px"])))
            self.table.setItem(
                row, 2, QTableWidgetItem(f"{m['equivalent_diameter_px']:.1f}")
            )
            self.table.setItem(row, 3, QTableWidgetItem(f"{m['eccentricity']:.2f}"))

    def export_csv(self) -> None:
        if not self.current_metrics:
            QMessageBox.information(self, "No data", "Run analysis first.")
            return
        default = (
            self.current_image_path.with_suffix(".csv")
            if self.current_image_path
            else Path("colonies.csv")
        )
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", str(default), "CSV (*.csv)"
        )
        if path_str:
            write_csv(self.current_metrics, Path(path_str))
            self.status_label.setText(f"Saved {path_str}")

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
