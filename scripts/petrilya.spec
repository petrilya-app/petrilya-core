# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Petrilya desktop app.

Run from repo root with:

    pyinstaller scripts/petrilya.spec

The resulting executable ends up in ``dist/petrilya.exe``.
"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------

ROOT = Path(SPECPATH).resolve().parent  # repo root (one above scripts/)
ENTRY = str(ROOT / "petrilya" / "ui" / "app.py")
ICON  = ROOT / "scripts" / "petrilya.ico"
ICON_PATH = str(ICON) if ICON.exists() else None

# ---------------------------------------------------------------------------
# Hidden imports — PyInstaller misses these when scanning the entry script
# ---------------------------------------------------------------------------

hiddenimports = []
hiddenimports += collect_submodules("petrilya")
hiddenimports += collect_submodules("cellpose")
hiddenimports += collect_submodules("skimage")
hiddenimports += collect_submodules("scipy")
hiddenimports += [
    # Often missed
    "scipy.special._cdflib",
    "scipy.special._ufuncs_cxx",
    "scipy.ndimage._morphology",
    "numpy.core._dtype_ctypes",
    "PySide6.QtSvg",
    "PIL.Image",
    "reportlab.graphics.barcode",
]

# ---------------------------------------------------------------------------
# Data files that ship inside the bundle
# ---------------------------------------------------------------------------

datas = []
# Cellpose ships per-model defaults, gui resources, etc.
datas += collect_data_files("cellpose")
# Scikit-image runtime resources
datas += collect_data_files("skimage")
# PIL fonts / metadata
datas += collect_data_files("PIL")
# Cyto3 model weights (if we pre-downloaded them into ./models/)
preshipped_weights = ROOT / "models" / "cyto3"
if preshipped_weights.exists():
    datas.append((str(preshipped_weights), "models"))

# ---------------------------------------------------------------------------
# Excludes — these explode the binary size and we don't need them
# ---------------------------------------------------------------------------

excludes = [
    "pytest", "tests", "IPython", "jupyter", "notebook", "ipykernel",
    "matplotlib.tests", "scipy.tests", "skimage.data._registry",
    "tkinter", "test",
]

# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

a = Analysis(
    [ENTRY],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="petrilya",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,             # compress; ~30% size reduction
    upx_exclude=[
        # UPX corrupts some Qt plugins — exclude common offenders
        "qwindows.dll", "Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll",
        "VCRUNTIME140.dll", "MSVCP140.dll",
    ],
    runtime_tmpdir=None,
    console=False,        # GUI app — no terminal
    icon=ICON_PATH,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
