"""Quick smoke test for engines on a real photo."""
import sys, time
import numpy as np
from PIL import Image
from petrilya.inference.engine import _find_dish_roi, _crop_to_dish, build_engine

path = sys.argv[1] if len(sys.argv) > 1 else "11764.jpg"
img = np.array(Image.open(path).convert("L"))
print(f"Image: {img.shape}, mean={img.mean():.0f}, std={img.std():.0f}")

roi = _find_dish_roi(img)
print(f"ROI (cx, cy, r): {roi}")

if roi is not None:
    crop, off = _crop_to_dish(img, roi)
    nz = crop > 0
    print(f"Crop: {crop.shape}, non-zero={nz.sum()}, off={off}")
    print(f"Crop[in]: min={crop[nz].min()}, max={crop.max()}, mean={crop[nz].mean():.0f}")
    from skimage import filters
    t = filters.threshold_otsu(crop[nz])
    bf = (crop[nz] > t).mean()
    polarity = "bright" if bf < 0.5 else "dark"
    print(f"Otsu={t}, bright_frac={bf:.2f}, polarity={polarity}")

print()
for name in ("classical", "cellpose-onnx"):
    print(f"=== {name} ===")
    try:
        eng = build_engine(name)
        t0 = time.perf_counter()
        masks, diam = eng.segment(img)
        dt_ms = (time.perf_counter() - t0) * 1000
        n = int(masks.max())
        print(f"  {dt_ms:.0f} ms, {n} colonies, diam~{diam:.1f}px")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
