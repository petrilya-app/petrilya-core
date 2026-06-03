"""Background workers for running inference off the UI thread."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from petrilya.export.csv_writer import write_csv
from petrilya.export.json_manifest import build_manifest, write_manifest
from petrilya.export.pdf_report import write_pdf_report
from petrilya.metrics.colony import compute_colony_metrics
from petrilya.ui.mock_engine import mock_segment


def run_segmentation(
    image: np.ndarray,
    use_gpu: bool,
) -> tuple[np.ndarray, float, str, dict]:
    """Run segmentation; returns (masks, elapsed, engine_name, params).

    Single integration point: swap mock_segment for CellposeEngine here
    once cellpose.org is back online.
    """
    t0 = time.perf_counter()
    masks, diam = mock_segment(image, n_colonies=180, delay_seconds=1.2)
    elapsed = time.perf_counter() - t0
    return masks, elapsed, "mock-v0", {"n_colonies": 180, "diameter_px": diam}


class WorkerSignals(QObject):
    started = Signal()
    progress = Signal(str)
    finished = Signal(object, object, list, float, str, dict)
    # image, masks, metrics, elapsed, engine_name, engine_params
    error = Signal(str)


class AnalysisWorker(QRunnable):
    """Runs segmentation + metrics on a background thread."""

    def __init__(
        self,
        image_path: Path,
        use_gpu: bool = False,
        scale_um_per_px: float | None = None,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.use_gpu = use_gpu
        self.scale_um_per_px = scale_um_per_px
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.started.emit()
            self.signals.progress.emit(f"Loading {self.image_path.name}...")
            image = np.array(Image.open(self.image_path).convert("L"))

            self.signals.progress.emit(
                f"Segmenting ({'GPU' if self.use_gpu else 'CPU'})..."
            )
            masks, elapsed, engine_name, params = run_segmentation(image, self.use_gpu)

            self.signals.progress.emit("Computing metrics...")
            metrics = compute_colony_metrics(masks, scale_um_per_px=self.scale_um_per_px)

            self.signals.finished.emit(
                image, masks, metrics, elapsed, engine_name, params
            )
        except Exception as e:  # noqa: BLE001
            self.signals.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


# ----------------------------- batch processing -----------------------------


class BatchSignals(QObject):
    progress = Signal(int, int, str)  # current, total, filename
    finished = Signal(int, int, Path)  # ok_count, fail_count, summary_csv
    error = Signal(str)


class BatchWorker(QRunnable):
    """Process every image in ``input_dir`` and write per-image artefacts.

    For each input image, writes:
      <name>.csv         per-colony metrics
      <name>.report.pdf  human-readable report
      <name>.manifest.json   reproducibility manifest

    Also writes ``summary.csv`` in the output directory with one row
    per image.
    """

    SUPPORTED = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        use_gpu: bool = False,
        scale_um_per_px: float | None = None,
    ) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.scale_um_per_px = scale_um_per_px
        self.signals = BatchSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            images = sorted(
                p
                for p in self.input_dir.iterdir()
                if p.suffix.lower() in self.SUPPORTED and p.is_file()
            )
            if not images:
                self.signals.error.emit(f"No supported images in {self.input_dir}")
                return

            summary_rows: list[dict] = []
            ok = 0
            fail = 0

            for i, img_path in enumerate(images, start=1):
                self.signals.progress.emit(i, len(images), img_path.name)
                try:
                    image = np.array(Image.open(img_path).convert("L"))
                    masks, elapsed, engine_name, params = run_segmentation(
                        image, self.use_gpu
                    )
                    metrics = compute_colony_metrics(
                        masks, scale_um_per_px=self.scale_um_per_px
                    )

                    base = self.output_dir / img_path.stem
                    write_csv(metrics, base.with_suffix(".csv"))
                    write_pdf_report(
                        base.with_suffix(".report.pdf"),
                        image=image,
                        masks=masks,
                        metrics=metrics,
                        image_name=img_path.name,
                        elapsed_seconds=elapsed,
                        engine_name=engine_name,
                        scale_um_per_px=self.scale_um_per_px,
                    )
                    manifest = build_manifest(
                        image_path=img_path,
                        masks_shape=masks.shape,
                        n_objects=len(metrics),
                        elapsed_seconds=elapsed,
                        engine_name=engine_name,
                        engine_params=params,
                        scale_um_per_px=self.scale_um_per_px,
                    )
                    write_manifest(manifest, base.with_suffix(".manifest.json"))

                    summary_rows.append(
                        {
                            "image": img_path.name,
                            "n_colonies": len(metrics),
                            "elapsed_seconds": round(elapsed, 3),
                            "engine": engine_name,
                        }
                    )
                    ok += 1
                except Exception as e:  # noqa: BLE001
                    fail += 1
                    summary_rows.append(
                        {
                            "image": img_path.name,
                            "n_colonies": -1,
                            "elapsed_seconds": -1,
                            "engine": f"ERROR: {type(e).__name__}: {e}",
                        }
                    )

            summary_path = self.output_dir / "summary.csv"
            import csv

            with summary_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f, fieldnames=["image", "n_colonies", "elapsed_seconds", "engine"]
                )
                w.writeheader()
                w.writerows(summary_rows)

            self.signals.finished.emit(ok, fail, summary_path)
        except Exception as e:  # noqa: BLE001
            self.signals.error.emit(
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            )
