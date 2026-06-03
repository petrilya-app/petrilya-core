"""Background worker for running inference off the UI thread."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from petrilya.metrics.colony import compute_colony_metrics
from petrilya.ui.mock_engine import mock_segment


class WorkerSignals(QObject):
    started = Signal()
    progress = Signal(str)
    finished = Signal(object, object, list, float)  # image, masks, metrics, elapsed
    error = Signal(str)


class AnalysisWorker(QRunnable):
    """Runs segmentation on a background thread.

    For now uses the mock engine. Swap in CellposeEngine once weights load.
    """

    def __init__(self, image_path: Path, use_gpu: bool = False) -> None:
        super().__init__()
        self.image_path = image_path
        self.use_gpu = use_gpu
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        import time

        try:
            self.signals.started.emit()
            self.signals.progress.emit(f"Loading {self.image_path.name}...")
            image = np.array(Image.open(self.image_path).convert("L"))

            self.signals.progress.emit(
                f"Segmenting ({'GPU' if self.use_gpu else 'CPU'})..."
            )
            t0 = time.perf_counter()
            masks, _diam = mock_segment(image, n_colonies=180, delay_seconds=1.2)
            elapsed = time.perf_counter() - t0

            self.signals.progress.emit("Computing metrics...")
            metrics = compute_colony_metrics(masks)

            self.signals.finished.emit(image, masks, metrics, elapsed)
        except Exception as e:  # noqa: BLE001
            self.signals.error.emit(f"{type(e).__name__}: {e}")
