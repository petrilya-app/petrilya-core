# Releasing Petrilya

There are two ways to ship a Windows `.exe`:

## Automatic — via a Git tag (recommended)

This is what powers the **Download for Windows** button on
[petrilya.com](https://petrilya.com).

1. Make sure `main` is in the state you want to ship.
2. Tag the commit with a semver tag:

   ```bash
   git tag v0.0.1
   git push origin v0.0.1
   ```

3. The `Build & release Windows binary` workflow
   ([`.github/workflows/release.yml`](.github/workflows/release.yml))
   picks up the tag, builds `petrilya-windows.exe` on `windows-latest`,
   and attaches it to a GitHub Release named `v0.0.1`.
4. After ~15-25 minutes the file is available at:

   ```
   https://github.com/petrilya-app/petrilya-core/releases/latest/download/petrilya-windows.exe
   ```

   That URL is permanent — every new release replaces the file behind it,
   so the landing-page button always serves the latest build.

You can watch the build at
[Actions → Build & release Windows binary](https://github.com/petrilya-app/petrilya-core/actions/workflows/release.yml).

## Manual local build

For testing changes before tagging, build the same binary on your machine:

```powershell
cd C:\path\to\petrilya-core
.\.venv\Scripts\Activate.ps1
python scripts/build_windows.py
```

Result: `dist/petrilya.exe` (~300-500 MB the first time, ~100-200 MB after UPX compression).

Run it to test:

```powershell
.\dist\petrilya.exe
```

## What's bundled

The build script puts everything the app needs inside the single `.exe`:

- Python 3.12 interpreter
- PySide6 + Qt runtime
- PyTorch CPU build
- Cellpose + scikit-image + scipy
- All Petrilya source code

What is **not** bundled (and is fetched at first launch):

- The Cellpose `cyto3` model weights — downloaded automatically from
  `cellpose.org` on the user's first analysis run (~25 MB). The app
  caches them under `%LOCALAPPDATA%\cellpose\models\`.

## Code signing (TODO)

Right now the binary is unsigned, so Windows SmartScreen will warn the
user with the "Unknown publisher" dialog on first launch. To remove that
warning we'd need:

- An EV (Extended Validation) code-signing certificate (~$300-600/year)
- Or an OV cert (cheaper but still triggers SmartScreen the first few
  times)
- Sign in CI with `signtool.exe` after the PyInstaller step

Plan: add this in v0.1 once we have a first batch of paid pilots.
