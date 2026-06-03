"""Command-line interface for Petrilya."""

from __future__ import annotations

import time
from pathlib import Path

import click
import numpy as np
from PIL import Image
from rich.console import Console

from petrilya.export.csv_writer import write_csv
from petrilya.inference.engine import CellposeEngine
from petrilya.metrics.colony import compute_colony_metrics

console = Console()


@click.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path), default=None)
@click.option("--gpu/--no-gpu", default=False, help="Use CUDA if available.")
@click.option(
    "--diameter",
    type=float,
    default=None,
    help="Approximate colony diameter in pixels (auto if omitted).",
)
def main(
    image_path: Path,
    output: Path | None,
    gpu: bool,
    diameter: float | None,
) -> None:
    """Analyze a petri dish image and export colony metrics."""
    output = output or image_path.with_suffix(".csv")

    console.print(f"[cyan]Loading[/] {image_path}")
    image = np.array(Image.open(image_path).convert("L"))
    console.print(f"  shape={image.shape} dtype={image.dtype}")

    console.print(f"[cyan]Initializing Cellpose[/] (gpu={gpu})")
    engine = CellposeEngine(use_gpu=gpu)

    t0 = time.perf_counter()
    masks, diam = engine.segment(image, diameter=diameter)
    elapsed = time.perf_counter() - t0

    metrics = compute_colony_metrics(masks)
    write_csv(metrics, output)

    console.print(
        f"[green]OK[/] Found [bold]{len(metrics)}[/] colonies "
        f"(diam~{diam:.1f}px) in [bold]{elapsed:.2f}s[/]"
    )
    console.print(f"[green]OK[/] Saved {output}")


if __name__ == "__main__":
    main()
