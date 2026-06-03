"""Benchmark inference speed on CPU vs GPU."""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from petrilya.inference.engine import CellposeEngine


def benchmark(image_path: Path, use_gpu: bool, n_runs: int = 5) -> dict:
    image = np.array(Image.open(image_path).convert("L"))
    engine = CellposeEngine(use_gpu=use_gpu)

    # warmup
    engine.segment(image)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        engine.segment(image)
        times.append(time.perf_counter() - t0)

    return {
        "shape": image.shape,
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "min": min(times),
        "max": max(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
    }


if __name__ == "__main__":
    img_arg = sys.argv[1] if len(sys.argv) > 1 else "tests/data/dish_001.jpg"
    img_path = Path(img_arg)
    if not img_path.exists():
        print(f"ERROR: {img_path} not found. Put a test image there first.")
        sys.exit(1)

    for gpu in [False, True]:
        try:
            r = benchmark(img_path, use_gpu=gpu, n_runs=3)
            print(
                f"GPU={gpu} shape={r['shape']}: "
                f"mean={r['mean']:.2f}s median={r['median']:.2f}s "
                f"min={r['min']:.2f}s max={r['max']:.2f}s"
            )
        except Exception as e:
            print(f"GPU={gpu}: SKIPPED ({type(e).__name__}: {e})")
