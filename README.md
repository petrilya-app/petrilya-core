# Petrilya

AI colony counter for biology labs — drag, count, export.

Open-source desktop application that turns a phone photo of a Petri dish into a clinical-grade colony count in seconds. No code, no cloud, no GPU required.

## Status

Pre-alpha. Sprint S0: technical prototype.

## Quick start (developer)

```powershell
git clone https://github.com/petrilya-app/petrilya-core.git
cd petrilya-core
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

# Optional: GPU acceleration (NVIDIA + CUDA 12)
pip install onnxruntime-gpu

# Run on a test image
petrilya path\to\dish.jpg --no-gpu
```

## Benchmarks

To be filled after S0 measurements.

| Machine | CPU | GPU | Image size | Time (s) | Colonies |
|---|---|---|---|---|---|
| TBD | TBD | RTX 4060 Ti (CUDA) | 1024x1024 | ? | ? |
| TBD | TBD | None (CPU) | 1024x1024 | ? | ? |

## License

[AGPL-3.0](LICENSE). Commercial license available — contact the maintainers.
