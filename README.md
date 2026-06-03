# Petrilya

**AI colony counter for biology labs — drag, count, export.**

Open-source desktop application that turns a phone photo of a Petri dish into a colony count, a CSV, and a one-page PDF report. No code, no cloud, no GPU required.

🌐 **Website:** [petrilya.com](https://petrilya.com) · 🐙 **Source:** [github.com/petrilya-app/petrilya-core](https://github.com/petrilya-app/petrilya-core)

---

## ⚠️ Pre-alpha

Petrilya is being built in public. The current build runs the full UI (drag-and-drop, zoom, mask editing, batch processing, CSV/PDF/JSON export) against a **mock segmentation engine** — real Cellpose integration is blocked on `cellpose.org` model-weight hosting being available again. Until then, the mock generates plausible-looking colony detections so the rest of the pipeline can be tested.

If you need a working colony counter today, you probably want [Cellpose](https://github.com/MouseLand/cellpose) directly. Petrilya's value-add is the no-code desktop wrapper, batch reports, and publication-ready outputs — coming soon.

---

## Install (developer)

```bash
git clone https://github.com/petrilya-app/petrilya-core.git
cd petrilya-core
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

Or, once published:

```bash
pip install petrilya
```

## Run the GUI

```bash
petrilya-ui
```

## CLI

```bash
petrilya path/to/dish.jpg --no-gpu
petrilya path/to/dish.jpg --gpu --diameter 30
```

---

## What works today

- ✅ Drag-and-drop image loading (JPG, PNG, TIFF, BMP)
- ✅ Zoomable, pannable canvas (mouse wheel, Space-drag)
- ✅ Mask overlay with adjustable opacity (0–100%)
- ✅ Manual mask editing: erase a click, paint with brush
- ✅ Per-colony metrics: area, diameter, eccentricity, solidity
- ✅ Micrometers-per-pixel scale → results in μm² and μm
- ✅ Batch processing of an entire folder
- ✅ CSV export (per-colony rows)
- ✅ PDF report (preview + histogram + table)
- ✅ JSON manifest with SHA-256 and parameters for reproducibility
- ⏳ Real Cellpose model — pending `cellpose.org` server recovery

## License

[**AGPL-3.0**](LICENSE) for the open-source build.
Commercial license available for organisations that need on-premise deployment without copyleft obligations — including GMP-relevant workflows. Contact `iliarogogvykh@gmail.com`.

## Acknowledgements

Built with [Cellpose](https://github.com/MouseLand/cellpose), [PySide6](https://doc.qt.io/qtforpython-6/), [ONNX Runtime](https://onnxruntime.ai/), [scikit-image](https://scikit-image.org/), and [ReportLab](https://www.reportlab.com/).
