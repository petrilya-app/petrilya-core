"""Build petrilya.exe locally (Windows).

Usage from the repo root::

    .\\.venv\\Scripts\\Activate.ps1
    python scripts/build_windows.py

The resulting file lands at ``dist/petrilya.exe``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    spec = repo_root / "scripts" / "petrilya.spec"

    print(f"[build] repo root: {repo_root}")
    print(f"[build] spec file: {spec}")

    # Make sure PyInstaller is available in the active interpreter
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        print("[build] PyInstaller not found, installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyinstaller>=6.0"]
        )

    # Generate an .ico if our SVG favicon is available and the .ico is missing
    icon_target = repo_root / "scripts" / "petrilya.ico"
    favicon = repo_root / "docs" / "favicon.svg"
    if not icon_target.exists() and favicon.exists():
        try:
            _svg_to_ico(favicon, icon_target)
            print(f"[build] generated icon: {icon_target}")
        except Exception as e:  # noqa: BLE001
            print(f"[build] icon generation failed (continuing without): {e}")

    # Wipe old build artefacts so PyInstaller doesn't cache stale data
    for d in ("build", "dist"):
        target = repo_root / d
        if target.exists():
            print(f"[build] cleaning {target}")
            import shutil
            shutil.rmtree(target, ignore_errors=True)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        str(spec),
    ]
    print(f"[build] running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(repo_root))

    out = repo_root / "dist" / "petrilya.exe"
    if out.exists():
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"\n[build] SUCCESS: {out}  ({size_mb:.1f} MB)")
        print(f"[build] try it: {out}")
    else:
        print("\n[build] ERROR: dist/petrilya.exe was not produced.")
        sys.exit(1)


def _svg_to_ico(svg_path: Path, ico_path: Path) -> None:
    """Convert an SVG into a multi-size .ico (16, 32, 48, 64, 128, 256 px)."""
    from PIL import Image
    try:
        import cairosvg
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "cairosvg"]
        )
        import cairosvg  # type: ignore[no-redef]
    import io

    png_bytes = cairosvg.svg2png(
        url=str(svg_path), output_width=256, output_height=256
    )
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    img.save(
        ico_path,
        format="ICO",
        sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )


if __name__ == "__main__":
    main()
