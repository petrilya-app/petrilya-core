"""CSV export for colony metrics."""

from __future__ import annotations

import csv
from pathlib import Path


CSV_HEADERS = [
    "id",
    "area_px",
    "centroid_y",
    "centroid_x",
    "equivalent_diameter_px",
    "eccentricity",
    "solidity",
]


def write_csv(metrics: list[dict], output_path: Path) -> None:
    """Write a list of metric dicts to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()
        writer.writerows(metrics)
