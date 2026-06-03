"""CSV export for colony metrics."""

from __future__ import annotations

import csv
from pathlib import Path


BASE_HEADERS = [
    "id",
    "area_px",
    "centroid_y",
    "centroid_x",
    "equivalent_diameter_px",
    "eccentricity",
    "solidity",
]
PHYSICAL_HEADERS = ["area_um2", "equivalent_diameter_um"]


def write_csv(metrics: list[dict], output_path: Path) -> None:
    """Write a list of metric dicts to CSV.

    Auto-detects whether physical-unit columns (um2, um) are present
    and includes them only if so.
    """
    headers = list(BASE_HEADERS)
    if metrics and "area_um2" in metrics[0]:
        headers.extend(PHYSICAL_HEADERS)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics)
