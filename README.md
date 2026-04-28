# MycoPrep

Bacterial microscopy pre-processing pipeline with a PyQt6 desktop GUI.

MycoPrep wraps focus picking, FOV split, Cellpose-SAM segmentation, and a cell-quality classifier into a single wizard-style application for fungal/bacterial specimen prep.

This repo merges the former **ImagingPipeline** (core library) and **ImagePipelineGUI** (PyQt6 frontend) projects into a single Python package: `mycoprep` with `mycoprep.core` (library) and `mycoprep.gui` (GUI) subpackages.

## Install

### For end-users (Windows)

1. Download the latest `MycoPrep-windows.zip` from the [Releases](https://github.com/timdewet/MycoPrep/releases) page.
2. Unzip anywhere (e.g. `C:\Program Files\MycoPrep\`).
3. Double-click `MycoPrep.exe`.

First launch downloads ~few hundred MB of Cellpose model weights to `%USERPROFILE%\.cellpose\` — needs internet once. Subsequent launches are offline. No Python install required.

### For developers (from source)

Works on macOS, Linux, and Windows.

```bash
git clone https://github.com/timdewet/MycoPrep.git
cd MycoPrep
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e .[test]
```

Requires Python 3.10+. PyTorch + Cellpose are heavy installs; first install may take a while.

## Update

### End-users (Windows zip)

1. Download the new `MycoPrep-windows.zip` from [Releases](https://github.com/timdewet/MycoPrep/releases).
2. Unzip over your existing `MycoPrep\` folder, replacing the old files. Cellpose weights in `%USERPROFILE%\.cellpose\` are kept.
3. Double-click `MycoPrep.exe`.

If Windows Explorer still shows the old icon after an update, move the folder once or run `ie4uinit.exe -show` to flush the icon cache.

### Developers (from source)

```bash
git pull
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -e .                  # re-runs if pyproject.toml or package-data changed
```

Re-run `pip install -e .` (not just `git pull`) whenever `pyproject.toml`, dependencies, or shipped data files (e.g. `src/mycoprep/core/models/`) change — editable installs don't auto-pick up new package-data entries.

### Developers rebuilding the Windows bundle

After `git pull` + `pip install -e .`:

```powershell
pyinstaller packaging\mycoprep.spec
```

Output is `dist\MycoPrep\`. Zip and ship as a new release.

## Run

| Command | Purpose |
| --- | --- |
| `mycoprep` | Launch the GUI |
| `mycoprep-cli --help` | Top-level CLI (was `imagingpipeline`) |
| `mycoprep-focus --help` | Focus-picking CLI (was `focuspicker`) |
| `python -m mycoprep` | Equivalent to `mycoprep` |

## Layout

```
src/mycoprep/
├── core/
│   ├── models/  # bundled classifier weights shipped via package-data
│   └── ...      # focus, segmentation, classification, CZI handling
└── gui/         # PyQt6 wizard: panels, pipeline runner, live preview
assets/
├── logo/        # app icon (svg + ico)
└── models_mtb/  # training artifacts (final_model.pth, curves, configs)
packaging/
└── mycoprep.spec  # PyInstaller spec for desktop bundle
```

## Building a desktop bundle

The bundle config lives in [packaging/mycoprep.spec](packaging/mycoprep.spec). PyInstaller does **not** cross-compile — build on the OS you intend to ship to.

### Windows (`.exe`)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
pip install pyinstaller
pyinstaller packaging\mycoprep.spec
```

Output: `dist\MycoPrep\` containing `MycoPrep.exe` plus DLLs/data. Zip the **whole folder** to distribute.

For GPU (CUDA) inference, install the matching PyTorch wheel before `pip install -e .`:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU-only is fine if you don't need GPU; the bundle is much smaller.

### macOS / Linux

```bash
pip install pyinstaller
pyinstaller packaging/mycoprep.spec
```

Output: `dist/MycoPrep/`. Runs only on the OS family it was built on.

### Building Windows from a non-Windows host

PyInstaller can't cross-compile. Use one of:

- **GitHub Actions** with `runs-on: windows-latest` (recommended for repeatable releases).
- A Windows VM (Parallels, UTM, VMware) on your Mac.
- A borrowed/spare Windows machine.

## Releases

Pre-built Windows bundles are published on the [Releases](https://github.com/timdewet/MycoPrep/releases) page. Each release ships a zipped `dist\MycoPrep\` folder — see the end-user install steps above.

## Acknowledgments

MycoPrep incorporates methods derived from previously published research:

- **Cellpose-SAM** for cell segmentation.
- **MicrobeJ** (Ducret et al., *Nat. Microbiol.* 2016), **Oufti** (Paintdakhi et al., *Mol. Microbiol.* 2016), and **PSICIC** (Guberman et al., *PLoS Comput. Biol.* 2008) for the sub-pixel contour / medial-axis / gradient-snap midline methodology.
- **MOMIA** ([jzrolling/MOMIA](https://github.com/jzrolling/MOMIA), MIT licence) as inspiration for the midline-derived per-cell morphology columns.

Our implementations are rewritten on top of modern scikit-image / SciPy primitives — no upstream code is vendored. Full per-module citations live alongside the code (see `src/mycoprep/core/extract/_midline.py`). Credit for the underlying methods belongs to their original authors.

## License

MIT — see [LICENSE](LICENSE).
