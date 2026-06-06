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
from petrilya.inference.engine import CellposeEngine
from petrilya.metrics.colony import compute_colony_metrics


def _load_image(path: Path) -> np.ndarray:
    """Load image preserving native colour mode for display."""
    pil = Image.open(path)
    if pil.mode in ("RGB", "RGBA", "L"):
        return np.array(pil)
    return np.array(pil.convert("RGB"))


_ENGINE_CACHE: dict[tuple, CellposeEngine] = {}


def _get_engine(use_gpu: bool) -> CellposeEngine:
    """Return a cached CellposeEngine, instantiating on first call.

    Building the engine loads ``cyto3`` weights and warms up PyTorch — a
    3-5 s hit each time. Without caching every Analyze click paid that
    cost; with the cache the model is built once per session per
    (use_gpu) variant and reused for every subsequent run.

    The cache is safe to share across QThreadPool workers because the
    UI disables the Analyze button while a run is in progress, so
    segment() is never called concurrently on the same engine.
    """
    key = (bool(use_gpu),)
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        engine = CellposeEngine(use_gpu=use_gpu)
        _ENGINE_CACHE[key] = engine
    return engine


def run_segmentation(
    image: np.ndarray,
    use_gpu: bool,
    dish_roi: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, float, str, dict]:
    """Run Cellpose segmentation; returns (masks, elapsed, name, manifest)."""
    t0 = time.perf_counter()
    engine = _get_engine(use_gpu)
    masks, _diam = engine.segment(image, dish_roi=dish_roi)
    elapsed = time.perf_counter() - t0
    manifest = engine.describe()
    if dish_roi is not None:
        manifest["params"]["dish_roi_manual"] = {
            "cx": float(dish_roi[0]),
            "cy": float(dish_roi[1]),
            "r":  float(dish_roi[2]),
        }
    return masks, elapsed, manifest["engine"], manifest


class WorkerSignals(QObject):
    started = Signal()
    progress = Signal(str)
    finished = Signal(object, object, list, float, str, dict)
    error = Signal(str)


class AnalysisWorker(QRunnable):
    def __init__(
        self,
        image_path: Path,
        use_gpu: bool = False,
        scale_um_per_px: float | None = None,
        dish_roi: tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self.image_path = image_path
        self.use_gpu = use_gpu
        self.scale_um_per_px = scale_um_per_px
        self.dish_roi = dish_roi
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.signals.started.emit()
            self.signals.progress.emit(f"Loading {self.image_path.name}...")
            display_image = _load_image(self.image_path)

            self.signals.progress.emit(
                f"Segmenting with Cellpose ({'GPU' if self.use_gpu else 'CPU'})..."
            )
            masks, elapsed, engine_name, params = run_segmentation(
                display_image, self.use_gpu, dish_roi=self.dish_roi
            )

            self.signals.progress.emit("Computing metrics...")
            metrics = compute_colony_metrics(masks, scale_um_per_px=self.scale_um_per_px)

            self.signals.finished.emit(
                display_image, masks, metrics, elapsed, engine_name, params
            )
        except Exception as e:  # noqa: BLE001
            self.signals.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


class BatchSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(int, int, Path)
    error = Signal(str)


class BatchWorker(QRunnable):
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
                    image = _load_image(img_path)
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
